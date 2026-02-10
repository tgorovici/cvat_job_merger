import streamlit as st
import xml.etree.ElementTree as ET
import json
import math
import zipfile
import tempfile
import shutil
from io import BytesIO
from copy import deepcopy
from collections import defaultdict
from pathlib import Path

st.set_page_config(page_title="CVAT / Datumaro Merger", layout="wide")
st.title("CVAT / Datumaro Track & Shape Merger (2D + 3D)")

st.markdown(
    """
Upload one of:
- **CVAT XML** (video track XML) → merge across segments/jobs, or merge “single-frame tracks” into one continuous track.
- **CVAT XML** (image/shapes XML) → convert per-image shapes into ONE track across the whole task.
- **CVAT ZIP** containing XML(s) → same as above (you choose which XML inside).
- **Datumaro JSON** or **Datumaro ZIP** (2D or 3D including `cuboid_3d`) → merge by label + track identity.

**Important:** In Datumaro, annotation `"id"` is a unique record id; track identity is `attributes.track_id` (and commonly `group`).
"""
)

uploaded = st.file_uploader("Upload .xml / .json / .zip", type=["xml", "json", "zip"])


# ============================================================
# Utility helpers
# ============================================================

def _try_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _export_xml_bytes(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)

def _next_track_id_xml(root: ET.Element) -> int:
    ids = []
    for t in root.findall("track"):
        v = _try_int(t.get("id", None))
        if v is not None:
            ids.append(v)
    return (max(ids) + 1) if ids else 0

def _safe_read_text(path: Path) -> str:
    # CVAT exports sometimes contain UTF-8, sometimes UTF-8 with BOM; try both
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="utf-8-sig")


# ============================================================
# ZIP handling (CVAT XML zips + Datumaro zips)
# ============================================================

def is_datumaro_json(data: dict) -> bool:
    return isinstance(data, dict) and "items" in data and "categories" in data and "label" in data["categories"]

def _find_datumaro_json_in_dir(root_dir: Path):
    for jf in sorted(root_dir.rglob("*.json")):
        try:
            data = json.loads(_safe_read_text(jf))
        except Exception:
            continue
        if is_datumaro_json(data):
            return jf, data
    return None, None

def _find_xml_files_in_dir(root_dir: Path):
    return sorted(root_dir.rglob("*.xml"))

def _extract_zip_to_tmp(zip_bytes: bytes) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="cvat_datumaro_zip_"))
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
        z.extractall(tmpdir)
    return tmpdir

def _rezip_folder_to_bytes(folder: Path) -> bytes:
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_dir():
                continue
            z.write(p, arcname=str(p.relative_to(folder)))
    bio.seek(0)
    return bio.getvalue()


# ============================================================
# CVAT VIDEO XML (track-based) helpers
# ============================================================

def parse_segments_xml(root):
    segments = []
    meta = root.find("meta")
    if meta is None:
        return segments

    task = meta.find("task")
    if task is not None:
        seg_container = task.find("segments")
        if seg_container is not None:
            for seg in seg_container.findall("segment"):
                seg_id = seg.findtext("id")
                start = seg.findtext("start")
                stop = seg.findtext("stop")
                if start is not None and stop is not None:
                    segments.append({
                        "id": seg_id if seg_id is not None else str(len(segments)),
                        "start": int(start),
                        "stop": int(stop),
                        "source": "segment",
                    })

    job = meta.find("job")
    if job is not None and not segments:
        start = job.findtext("start_frame")
        stop = job.findtext("stop_frame")
        if start is not None and stop is not None:
            segments.append({
                "id": job.findtext("id") or "0",
                "start": int(start),
                "stop": int(stop),
                "source": "job",
            })

    return segments

def fallback_single_segment_from_track_xml(root):
    frames = []
    for tr in root.findall("track"):
        for node in list(tr):
            if node.tag == "attribute":
                continue
            fr = _try_int(node.get("frame", None))
            if fr is not None:
                frames.append(fr)
    if not frames:
        return [{"id": "0", "start": 0, "stop": 0, "source": "fallback"}]
    return [{"id": "0", "start": min(frames), "stop": max(frames), "source": "fallback"}]

def collect_job_tracks_xml(root, segments):
    job_tracks = {seg["id"]: {} for seg in segments}

    for track in root.findall("track"):
        tid = track.get("id")
        label = track.get("label", "")

        attrs = {}
        for a in track.findall("attribute"):
            name = a.get("name")
            if name:
                attrs[name] = (a.text or "").strip()

        for node in list(track):
            if node.tag == "attribute":
                continue
            fr = _try_int(node.get("frame", None))
            if fr is None:
                continue
            for seg in segments:
                if seg["start"] <= fr <= seg["stop"]:
                    seg_tracks = job_tracks.setdefault(seg["id"], {})
                    info = seg_tracks.setdefault(
                        tid,
                        {"label": label, "attributes": attrs, "frames": []},
                    )
                    info["frames"].append(fr)

    return job_tracks

def _track_centers_for_boxes(track):
    out = []
    for box in track.findall("box"):
        try:
            frame = int(box.get("frame", 0))
            xtl = float(box.get("xtl", 0))
            ytl = float(box.get("ytl", 0))
            xbr = float(box.get("xbr", 0))
            ybr = float(box.get("ybr", 0))
        except Exception:
            continue
        cx = (xtl + xbr) / 2.0
        cy = (ytl + ybr) / 2.0
        out.append((frame, cx, cy))
    out.sort(key=lambda x: x[0])
    return out

def _attr_signature(attrs):
    return tuple(sorted(attrs.items()))

def _euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def build_track_info_xml(root, segments, job_tracks):
    track_map = {t.get("id"): t for t in root.findall("track")}
    info = {seg["id"]: {} for seg in segments}

    for seg in segments:
        sid = seg["id"]
        for tid, meta in job_tracks.get(sid, {}).items():
            t = track_map.get(tid)
            if t is None:
                continue

            centers = _track_centers_for_boxes(t)
            if centers:
                frames = [c[0] for c in centers]
                info[sid][tid] = {
                    "label": meta["label"],
                    "attributes": meta["attributes"],
                    "start_frame": min(frames),
                    "end_frame": max(frames),
                    "first_box": centers[0],
                    "last_box": centers[-1],
                    "has_boxes": True,
                }
            else:
                frames = []
                for node in list(t):
                    if node.tag == "attribute":
                        continue
                    fr = _try_int(node.get("frame", None))
                    if fr is not None:
                        frames.append(fr)
                if not frames:
                    continue
                info[sid][tid] = {
                    "label": meta["label"],
                    "attributes": meta["attributes"],
                    "start_frame": min(frames),
                    "end_frame": max(frames),
                    "first_box": None,
                    "last_box": None,
                    "has_boxes": False,
                }
    return info

def suggest_track_chain_xml(segments, track_info):
    suggestion = {}
    prev_tid = None
    prev_sid = None

    for seg in sorted(segments, key=lambda s: s["start"]):
        sid = seg["id"]
        candidates = track_info.get(sid, {})
        if not candidates:
            continue

        if len(candidates) == 1:
            tid = next(iter(candidates.keys()))
            suggestion[sid] = [tid]
            prev_tid, prev_sid = tid, sid
            continue

        if prev_tid is None or prev_sid is None:
            best_tid = max(
                candidates.items(),
                key=lambda kv: kv[1]["end_frame"] - kv[1]["start_frame"]
            )[0]
            suggestion[sid] = [best_tid]
            prev_tid, prev_sid = best_tid, sid
            continue

        prev_info = track_info.get(prev_sid, {}).get(prev_tid)
        if not prev_info or not prev_info.get("has_boxes") or not prev_info.get("last_box"):
            best_tid = max(
                candidates.items(),
                key=lambda kv: kv[1]["end_frame"] - kv[1]["start_frame"]
            )[0]
            suggestion[sid] = [best_tid]
            prev_tid, prev_sid = best_tid, sid
            continue

        prev_label = prev_info["label"]
        prev_attr_sig = _attr_signature(prev_info["attributes"])
        prev_center = (prev_info["last_box"][1], prev_info["last_box"][2])

        best_tid = None
        best_score = float("inf")

        for tid, ci in candidates.items():
            score = 0.0
            if ci["label"] != prev_label:
                score += 10_000
            if _attr_signature(ci["attributes"]) != prev_attr_sig:
                score += 1_000

            if ci.get("has_boxes") and ci.get("first_box"):
                cand_center = (ci["first_box"][1], ci["first_box"][2])
                score += _euclidean(prev_center, cand_center)
            else:
                score += 50_000

            if score < best_score:
                best_score = score
                best_tid = tid

        if best_tid is None:
            best_tid = max(
                candidates.items(),
                key=lambda kv: kv[1]["end_frame"] - kv[1]["start_frame"]
            )[0]

        suggestion[sid] = [best_tid]
        prev_tid, prev_sid = best_tid, sid

    return suggestion

def build_selection_ui_xml(segments, job_tracks, track_info, auto_suggestion):
    st.subheader("Track selection per segment/job (you can override suggestions)")
    selected = {}

    for seg in sorted(segments, key=lambda s: s["start"]):
        sid = seg["id"]
        tracks = job_tracks.get(sid, {})
        if not tracks:
            continue

        st.markdown(f"**Segment / Job {sid}** (frames {seg['start']}–{seg['stop']})")

        options = []
        option_to_tid = {}
        default = []

        for tid, meta in tracks.items():
            ti = track_info.get(sid, {}).get(tid, {})
            attr_str = ", ".join(f"{k}={v}" for k, v in meta["attributes"].items()) or "no attributes"
            sf, ef = ti.get("start_frame"), ti.get("end_frame")
            fr_str = f"{sf}-{ef}" if sf is not None and ef is not None else "n/a"
            desc = f"Label: {meta['label']} | Track ID: {tid} | {attr_str} | Frames: {fr_str}"
            if auto_suggestion.get(sid) == [tid]:
                desc += "  ⟵ suggested"
                default = [desc]
            options.append(desc)
            option_to_tid[desc] = tid

        chosen = st.multiselect(
            f"Choose track(s) in segment {sid} to be part of the SAME merged chain:",
            options,
            default=default or options[0:1],
            key=f"xml_seg_{sid}",
        )
        selected[sid] = [option_to_tid[c] for c in chosen]

    return selected

def merge_tracks_xml_by_segments(root, segments, selected_tracks):
    new_id = _next_track_id_xml(root)
    track_map = {t.get("id"): t for t in root.findall("track")}

    new_track = ET.Element("track")
    new_track.set("id", str(new_id))

    base_label = None
    base_group = None
    base_source = None
    base_attrs = []
    used_track_ids = set()

    for seg in sorted(segments, key=lambda s: s["start"]):
        sid = seg["id"]
        tids = selected_tracks.get(sid, [])
        if not tids:
            continue

        for tid in tids:
            orig = track_map.get(tid)
            if orig is None:
                continue
            used_track_ids.add(tid)

            if base_label is None:
                base_label = orig.get("label", "")
                base_group = orig.get("group", "0")
                base_source = orig.get("source", "manual")
                for a in orig.findall("attribute"):
                    base_attrs.append(deepcopy(a))

            for node in list(orig):
                if node.tag == "attribute":
                    continue
                frame = _try_int(node.get("frame", None))
                if frame is None:
                    continue
                if seg["start"] <= frame <= seg["stop"]:
                    new_track.append(deepcopy(node))

    if len(list(new_track)) == 0:
        return None

    if base_label is not None:
        new_track.set("label", base_label)
    if base_group is not None:
        new_track.set("group", base_group)
    if base_source is not None:
        new_track.set("source", base_source)

    for a in reversed(base_attrs):
        new_track.insert(0, a)

    geom = [n for n in list(new_track) if n.tag != "attribute"]
    geom_sorted = sorted(geom, key=lambda n: int(n.get("frame", 0)))
    for n in geom:
        new_track.remove(n)
    for n in geom_sorted:
        new_track.append(n)

    for tid in used_track_ids:
        t = track_map.get(tid)
        if t is not None and t in list(root):
            root.remove(t)

    root.append(new_track)
    return root

def _box_area_node(box_node):
    try:
        xtl = float(box_node.get("xtl", 0))
        ytl = float(box_node.get("ytl", 0))
        xbr = float(box_node.get("xbr", 0))
        ybr = float(box_node.get("ybr", 0))
        return max(0.0, xbr - xtl) * max(0.0, ybr - ytl)
    except Exception:
        return 0.0

def merge_single_frame_tracks_into_one_track_xml(
    root: ET.Element,
    label_name: str,
    shape_tag: str = "box",
    resolve_same_frame: str = "largest",  # largest / first
    keep_attributes_from: str = "first",  # first / none
):
    tracks = [t for t in root.findall("track") if t.get("label", "") == label_name]
    if not tracks:
        return None, 0, "No tracks found for that label."

    new_id = _next_track_id_xml(root)

    per_frame = defaultdict(list)
    for t in tracks:
        for node in t.findall(shape_tag):
            fr = _try_int(node.get("frame", None))
            if fr is None:
                continue
            per_frame[fr].append(deepcopy(node))

    if not per_frame:
        return None, 0, f"No <{shape_tag}> nodes with frame=... found in tracks for label '{label_name}'."

    chosen = {}
    for fr, nodes in per_frame.items():
        if len(nodes) == 1:
            chosen[fr] = nodes[0]
        else:
            if shape_tag == "box" and resolve_same_frame == "largest":
                chosen[fr] = max(nodes, key=_box_area_node)
            else:
                chosen[fr] = nodes[0]

    new_track = ET.Element("track", {"id": str(new_id), "label": label_name, "source": "manual", "group": "0"})
    if keep_attributes_from == "first":
        for a in tracks[0].findall("attribute"):
            new_track.append(deepcopy(a))

    for fr in sorted(chosen.keys()):
        new_track.append(chosen[fr])

    for t in tracks:
        root.remove(t)
    root.append(new_track)

    return root, len(tracks), f"Merged {len(tracks)} tracks → 1 track with {len(chosen)} frames."

# ============================================================
# CVAT IMAGE/SHAPES XML -> ONE TRACK conversion
# ============================================================

def shapes_to_single_track_xml(
    root: ET.Element,
    shape_tag: str,
    label_name: str,
    resolve_same_frame: str = "largest",  # largest/first
    remove_original_shapes: bool = True,
):
    images = root.findall("image")
    if not images:
        return None, "No <image> elements found."

    per_frame = defaultdict(list)
    for img in images:
        frame = _try_int(img.get("id", "0"), default=0)
        for node in img.findall(shape_tag):
            if node.get("label") == label_name:
                per_frame[frame].append(node)

    if not per_frame:
        return None, f"No <{shape_tag}> shapes found for label '{label_name}'."

    chosen = {}
    for frame, nodes in per_frame.items():
        if len(nodes) == 1:
            chosen[frame] = nodes[0]
        else:
            if shape_tag == "box" and resolve_same_frame == "largest":
                chosen[frame] = max(nodes, key=_box_area_node)
            else:
                chosen[frame] = nodes[0]

    new_id = _next_track_id_xml(root)
    new_track = ET.Element("track", {"id": str(new_id), "label": label_name, "source": "manual", "group": "0"})

    for frame in sorted(chosen.keys()):
        src = chosen[frame]
        attrib = dict(src.attrib)

        base = {
            "frame": str(frame),
            "outside": "0",
            "occluded": attrib.get("occluded", "0"),
            "keyframe": "1",
            "z_order": attrib.get("z_order", "0"),
        }

        if shape_tag == "box":
            base.update({
                "xtl": attrib["xtl"],
                "ytl": attrib["ytl"],
                "xbr": attrib["xbr"],
                "ybr": attrib["ybr"],
            })
            dst = ET.Element("box", base)

        elif shape_tag in ("polygon", "polyline", "points"):
            if "points" not in attrib:
                continue
            base.update({"points": attrib["points"]})
            dst = ET.Element(shape_tag, base)

        else:
            return None, f"Unsupported shape tag: {shape_tag}"

        for a in src.findall("attribute"):
            dst.append(deepcopy(a))

        new_track.append(dst)

    if remove_original_shapes:
        for img in images:
            to_remove = []
            for node in img.findall(shape_tag):
                if node.get("label") == label_name:
                    to_remove.append(node)
            for node in to_remove:
                img.remove(node)

    root.append(new_track)
    return root, f"Created 1 track from shapes: label='{label_name}', frames={len(chosen)}."


# ============================================================
# Datumaro helpers (JSON + ZIP)
# ============================================================

def datumaro_label_map(data: dict):
    labels = data["categories"]["label"].get("labels", [])
    return {i: lbl.get("name", f"label_{i}") for i, lbl in enumerate(labels)}

def collect_datumaro_tracks(data: dict):
    label_map = datumaro_label_map(data)
    tracks = defaultdict(lambda: defaultdict(lambda: {"count": 0, "min_frame": None, "max_frame": None, "types": set()}))

    for item in data.get("items", []):
        frame = item.get("attr", {}).get("frame", None)
        for ann in item.get("annotations", []):
            attrs = ann.get("attributes", {}) or {}
            if "track_id" not in attrs:
                continue
            tid = attrs["track_id"]
            label_id = ann.get("label_id", -1)
            label_name = label_map.get(label_id, f"label_{label_id}")
            atype = ann.get("type", "unknown")

            rec = tracks[label_name][tid]
            rec["count"] += 1
            rec["types"].add(atype)
            if isinstance(frame, int):
                rec["min_frame"] = frame if rec["min_frame"] is None else min(rec["min_frame"], frame)
                rec["max_frame"] = frame if rec["max_frame"] is None else max(rec["max_frame"], frame)

    return tracks

def datumaro_merge_selected_track_ids(
    data: dict,
    label_name: str,
    source_track_ids,
    target_track_id: int,
    set_group: bool = True,
    add_instance_id: bool = True,
    reindex_annotation_ids: bool = False,
):
    labels = data["categories"]["label"].get("labels", [])
    label_map = {i: lbl.get("name", f"label_{i}") for i, lbl in enumerate(labels)}
    label_ids_for_name = {lid for lid, nm in label_map.items() if nm == label_name}

    source_set = set(source_track_ids)
    changed = 0

    for item in data.get("items", []):
        for ann in item.get("annotations", []):
            if ann.get("label_id") not in label_ids_for_name:
                continue
            attrs = ann.get("attributes", {}) or {}
            if attrs.get("track_id") in source_set:
                attrs["track_id"] = int(target_track_id)
                if add_instance_id:
                    attrs["instance_id"] = int(target_track_id)
                ann["attributes"] = attrs
                if set_group:
                    ann["group"] = int(target_track_id)
                changed += 1

    if reindex_annotation_ids:
        new_id = 0
        for item in data.get("items", []):
            for ann in item.get("annotations", []):
                ann["id"] = new_id
                new_id += 1

    return data, changed

def datumaro_merge_all_shapes_to_one_track(
    data: dict,
    label_name: str,
    target_track_id: int,
    force_add_track_id: bool = True,
    set_group: bool = True,
    add_instance_id: bool = True,
    reindex_annotation_ids: bool = False,
):
    labels = data["categories"]["label"].get("labels", [])
    label_map = {i: lbl.get("name", f"label_{i}") for i, lbl in enumerate(labels)}
    label_ids_for_name = {lid for lid, nm in label_map.items() if nm == label_name}

    changed = 0
    for item in data.get("items", []):
        for ann in item.get("annotations", []):
            if ann.get("label_id") not in label_ids_for_name:
                continue
            attrs = ann.get("attributes", {}) or {}

            if "track_id" not in attrs and not force_add_track_id:
                continue

            attrs["track_id"] = int(target_track_id)
            if add_instance_id:
                attrs["instance_id"] = int(target_track_id)
            ann["attributes"] = attrs
            if set_group:
                ann["group"] = int(target_track_id)
            changed += 1

    if reindex_annotation_ids:
        new_id = 0
        for item in data.get("items", []):
            for ann in item.get("annotations", []):
                ann["id"] = new_id
                new_id += 1

    return data, changed


# ============================================================
# Main loader: supports .zip with either CVAT XML(s) or Datumaro
# ============================================================

if uploaded is None:
    st.stop()

tmpdir = None  # for zip
zip_selected_path = None  # if user picks an xml/json inside zip
zip_mode = False
input_kind = None  # "cvat_xml" / "datumaro" / None

root_xml = None
datumaro_data = None

fname = uploaded.name.lower()

try:
    if fname.endswith(".zip"):
        zip_mode = True
        raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        tmpdir = _extract_zip_to_tmp(raw)

        # Try Datumaro first
        dm_json_path, dm_data = _find_datumaro_json_in_dir(tmpdir)

        # Find XML files
        xml_files = _find_xml_files_in_dir(tmpdir)

        # Let user choose which asset to use if ambiguous
        choices = []
        if dm_json_path is not None:
            choices.append(("Datumaro (auto-found JSON)", dm_json_path))
        for xf in xml_files:
            choices.append((f"CVAT XML: {xf.relative_to(tmpdir)}", xf))

        if not choices:
            st.error("ZIP extracted, but found no Datumaro JSON and no XML files.")
            st.stop()

        label_choices = [c[0] for c in choices]
        pick = st.selectbox("ZIP contains multiple candidates — choose what to load", label_choices)

        chosen_path = dict(choices)[pick]
        zip_selected_path = chosen_path

        if "Datumaro" in pick:
            input_kind = "datumaro"
            datumaro_data = dm_data
        else:
            input_kind = "cvat_xml"
            # parse xml from file path
            try:
                xml_text = _safe_read_text(chosen_path)
                root_xml = ET.fromstring(xml_text)
            except Exception as e:
                st.error(f"Failed to parse selected XML inside ZIP: {e}")
                st.stop()

    elif fname.endswith(".xml"):
        input_kind = "cvat_xml"
        tree = ET.parse(uploaded)
        root_xml = tree.getroot()

    elif fname.endswith(".json"):
        data = json.load(uploaded)
        if not is_datumaro_json(data):
            st.error("This JSON does not look like Datumaro (missing items/categories/label).")
            st.stop()
        input_kind = "datumaro"
        datumaro_data = data

    else:
        st.error("Unsupported file type. Upload .xml / .json / .zip")
        st.stop()

except Exception as e:
    st.error(f"Failed to load input: {e}")
    st.stop()


# ============================================================
# CVAT XML UI (track XML + shapes XML + special per-frame tracks merge)
# ============================================================

if input_kind == "cvat_xml":
    st.header("Mode: CVAT XML")

    has_images = len(root_xml.findall("image")) > 0
    has_tracks = len(root_xml.findall("track")) > 0

    tabs = st.tabs(["Auto Detect", "Video Track XML", "Image/Shapes XML → Track", "Per-frame tracks → ONE track"])

    with tabs[0]:
        if has_images and not has_tracks:
            st.info("Detected **image/shapes-style** CVAT XML (has <image>, no <track>). Use Shapes→Track tab.")
        elif has_tracks:
            st.info("Detected **track-style** CVAT XML (has <track>). Use Video Track XML or Per-frame tab.")
        else:
            st.warning("Could not clearly detect. Try the tabs manually.")

    with tabs[1]:
        if not has_tracks:
            st.warning("This XML does not contain <track> elements.")
        else:
            st.subheader("A) Merge selected tracks across segments/jobs into ONE track")
            segments = parse_segments_xml(root_xml)
            if not segments:
                segments = fallback_single_segment_from_track_xml(root_xml)
                st.info("No jobs/segments found — using a single fallback segment spanning all frames.")

            job_tracks = collect_job_tracks_xml(root_xml, segments)
            track_info = build_track_info_xml(root_xml, segments, job_tracks)
            auto_suggestion = suggest_track_chain_xml(segments, track_info)

            st.caption("Detected segments/jobs:")
            for seg in sorted(segments, key=lambda s: s["start"]):
                sid = seg["id"]
                tracks = job_tracks.get(sid, {})
                st.write(f"- Segment/Job {sid}: frames {seg['start']}–{seg['stop']} | tracks found: {len(tracks)}")

            selected_tracks = build_selection_ui_xml(segments, job_tracks, track_info, auto_suggestion)

            out_as_zip = False
            if zip_mode:
                out_as_zip = st.checkbox("Download result as ZIP (replace the selected XML inside the ZIP)", value=True)

            if st.button("Generate merged XML (ONE track from selections)", type="primary"):
                merged_root = merge_tracks_xml_by_segments(deepcopy(root_xml), segments, selected_tracks)
                if merged_root is None:
                    st.error("No valid merged track built. Check selections.")
                else:
                    st.success("Merged track created from your selections.")

                    merged_xml_bytes = _export_xml_bytes(merged_root)

                    if zip_mode and out_as_zip and tmpdir and zip_selected_path:
                        # replace the chosen xml inside tmpdir
                        zip_selected_path.write_bytes(merged_xml_bytes)
                        out_zip = _rezip_folder_to_bytes(tmpdir)
                        st.download_button(
                            "Download merged ZIP",
                            data=out_zip,
                            file_name="merged_cvat.zip",
                            mime="application/zip",
                        )
                    else:
                        st.download_button(
                            "Download merged CVAT XML",
                            data=merged_xml_bytes,
                            file_name="cvat_merged_track.xml",
                            mime="application/xml",
                        )

    with tabs[2]:
        if not has_images:
            st.warning("This XML does not contain <image> elements (not shapes-style).")
        else:
            st.subheader("Convert per-image shapes into ONE track across the whole task")

            shape_tags = ["box", "polygon", "polyline", "points"]
            available = []
            for tag in shape_tags:
                for img in root_xml.findall("image"):
                    if img.find(tag) is not None:
                        available.append(tag)
                        break

            if not available:
                st.warning("No supported shapes found inside <image> (box/polygon/polyline/points).")
            else:
                shape_tag = st.selectbox("Shape type to convert", available)

                labels = sorted({
                    n.get("label", "")
                    for img in root_xml.findall("image")
                    for n in img.findall(shape_tag)
                    if n.get("label", "")
                })

                if not labels:
                    st.warning(f"No labels found for <{shape_tag}> shapes.")
                else:
                    label_name = st.selectbox("Label to convert into a track", labels)
                    resolve = st.selectbox("If multiple shapes exist on the same frame, keep:", ["largest", "first"])
                    remove_orig = st.checkbox("Remove original shapes after creating track", value=True)

                    out_as_zip = False
                    if zip_mode:
                        out_as_zip = st.checkbox("Download result as ZIP (replace the selected XML inside the ZIP)", value=True, key="zip_shapes_replace")

                    if st.button("Convert shapes → ONE track", type="primary"):
                        merged_root, msg = shapes_to_single_track_xml(
                            deepcopy(root_xml),
                            shape_tag=shape_tag,
                            label_name=label_name,
                            resolve_same_frame=resolve,
                            remove_original_shapes=remove_orig,
                        )
                        if merged_root is None:
                            st.error(msg)
                        else:
                            st.success(msg)
                            merged_xml_bytes = _export_xml_bytes(merged_root)

                            if zip_mode and out_as_zip and tmpdir and zip_selected_path:
                                zip_selected_path.write_bytes(merged_xml_bytes)
                                out_zip = _rezip_folder_to_bytes(tmpdir)
                                st.download_button(
                                    "Download merged ZIP",
                                    data=out_zip,
                                    file_name="merged_cvat.zip",
                                    mime="application/zip",
                                )
                            else:
                                st.download_button(
                                    "Download XML with new track",
                                    data=merged_xml_bytes,
                                    file_name=f"cvat_shapes_to_track_{shape_tag}_{label_name}.xml",
                                    mime="application/xml",
                                )

    with tabs[3]:
        if not has_tracks:
            st.warning("This XML does not contain <track> elements.")
        else:
            st.subheader("Merge “one-track-per-frame” tracks into ONE continuous track")

            labels = sorted({t.get("label", "") for t in root_xml.findall("track") if t.get("label", "")})
            if not labels:
                st.warning("No labels found in tracks.")
            else:
                label_name = st.selectbox("Label to merge into one continuous track", labels, key="sf_label")
                resolve = st.selectbox("If multiple boxes exist on same frame, keep:", ["largest", "first"], key="sf_resolve")
                keep_attrs = st.selectbox("Track attributes to keep", ["first", "none"], key="sf_keep_attrs")

                out_as_zip = False
                if zip_mode:
                    out_as_zip = st.checkbox("Download result as ZIP (replace the selected XML inside the ZIP)", value=True, key="zip_sf_replace")

                if st.button("Merge per-frame tracks → ONE track", type="primary", key="sf_btn"):
                    merged_root, count, msg = merge_single_frame_tracks_into_one_track_xml(
                        deepcopy(root_xml),
                        label_name=label_name,
                        shape_tag="box",
                        resolve_same_frame=resolve,
                        keep_attributes_from=keep_attrs,
                    )
                    if merged_root is None:
                        st.error(msg)
                    else:
                        st.success(msg)
                        merged_xml_bytes = _export_xml_bytes(merged_root)

                        if zip_mode and out_as_zip and tmpdir and zip_selected_path:
                            zip_selected_path.write_bytes(merged_xml_bytes)
                            out_zip = _rezip_folder_to_bytes(tmpdir)
                            st.download_button(
                                "Download merged ZIP",
                                data=out_zip,
                                file_name="merged_cvat.zip",
                                mime="application/zip",
                            )
                        else:
                            st.download_button(
                                "Download merged CVAT Video XML",
                                data=merged_xml_bytes,
                                file_name=f"cvat_merge_singleframe_{label_name}.xml",
                                mime="application/xml",
                            )

# ============================================================
# Datumaro UI
# ============================================================

elif input_kind == "datumaro":
    st.header("Mode: Datumaro (2D/3D)")

    label_map = datumaro_label_map(datumaro_data)
    all_label_names = sorted(set(label_map.values()))

    tracks = collect_datumaro_tracks(datumaro_data)

    tabs = st.tabs(["Merge selected track_ids", "Merge ALL shapes into ONE track_id (whole task)"])

    with tabs[0]:
        st.subheader("Merge selected track_ids within a label")

        label_choice = st.selectbox("Label/class", all_label_names, key="dm_label_1")

        tid_options = []
        tid_lookup = {}
        if label_choice in tracks and tracks[label_choice]:
            for tid, rec in sorted(tracks[label_choice].items(), key=lambda kv: kv[0]):
                fr = "n/a"
                if rec["min_frame"] is not None and rec["max_frame"] is not None:
                    fr = f"{rec['min_frame']}-{rec['max_frame']}"
                types = ", ".join(sorted(rec["types"])) if rec["types"] else "unknown"
                disp = f"Label: {label_choice} | track_id: {tid} | frames: {fr} | anns: {rec['count']} | types: {types}"
                tid_lookup[disp] = tid
                tid_options.append(disp)

        if not tid_options:
            st.warning("No track_id found for this label. Use the 'Merge ALL shapes' tab if you want to force one track_id.")
        else:
            selected_disps = st.multiselect(
                "Select track_ids to merge",
                tid_options,
                default=tid_options[: min(3, len(tid_options))],
                key="dm_tids_multi",
            )
            selected_tids = [tid_lookup[d] for d in selected_disps]

            colA, colB, colC = st.columns([2, 2, 2])
            with colA:
                target_mode = st.radio("Target track_id", ["Use smallest selected", "Enter manually"], horizontal=True)
            with colB:
                set_group = st.checkbox("Unify 'group' to target track_id", value=True)
            with colC:
                add_instance_id = st.checkbox("Add attributes.instance_id = target track_id", value=True)

            target_tid = None
            if target_mode == "Enter manually":
                target_tid = st.number_input("Target track_id (int)", min_value=0, value=int(min(selected_tids)) if selected_tids else 0, step=1)

            reindex_ids = st.checkbox("Reindex annotation 'id' fields (unique 0..N-1)", value=False)

            out_as_zip = False
            if zip_mode:
                out_as_zip = st.checkbox("Download result as ZIP (replace Datumaro JSON inside ZIP)", value=True, key="dm_zip_replace_1")

            if st.button("Merge selected track_ids", type="primary"):
                if len(selected_tids) < 2:
                    st.error("Select at least 2 track_ids to merge.")
                else:
                    tgt = int(min(selected_tids)) if target_mode == "Use smallest selected" else int(target_tid)
                    merged, changed = datumaro_merge_selected_track_ids(
                        deepcopy(datumaro_data),
                        label_name=label_choice,
                        source_track_ids=selected_tids,
                        target_track_id=tgt,
                        set_group=set_group,
                        add_instance_id=add_instance_id,
                        reindex_annotation_ids=reindex_ids,
                    )
                    st.success(f"Updated {changed} annotations: merged {len(selected_tids)} track_ids → {tgt} for label '{label_choice}'.")

                    out_json = json.dumps(merged, ensure_ascii=False, indent=2).encode("utf-8")

                    if zip_mode and out_as_zip and tmpdir:
                        # replace the first Datumaro JSON we found
                        dm_json_path, _ = _find_datumaro_json_in_dir(tmpdir)
                        if dm_json_path is None:
                            st.warning("Could not re-find Datumaro JSON to replace; downloading JSON instead.")
                            st.download_button("Download merged Datumaro JSON", data=out_json, file_name="merged_datumaro.json", mime="application/json")
                        else:
                            dm_json_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
                            out_zip = _rezip_folder_to_bytes(tmpdir)
                            st.download_button("Download merged Datumaro ZIP", data=out_zip, file_name="merged_datumaro.zip", mime="application/zip")
                    else:
                        st.download_button("Download merged Datumaro JSON", data=out_json, file_name="merged_datumaro.json", mime="application/json")

    with tabs[1]:
        st.subheader("Merge ALL shapes of a label into ONE track_id (whole task)")

        label_choice2 = st.selectbox("Label/class", all_label_names, key="dm_label_2")

        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            target_track_id = st.number_input("Target track_id", min_value=0, value=0, step=1, key="dm_all_target")
        with col2:
            force_add_track_id = st.checkbox("If an annotation lacks track_id, add it", value=True, key="dm_all_force")
        with col3:
            set_group2 = st.checkbox("Unify 'group' to target track_id", value=True, key="dm_all_group")

        col4, col5 = st.columns([2, 2])
        with col4:
            add_instance_id2 = st.checkbox("Add attributes.instance_id = target track_id", value=True, key="dm_all_instance")
        with col5:
            reindex_ids2 = st.checkbox("Reindex annotation 'id' fields (unique 0..N-1)", value=False, key="dm_all_reindex")

        out_as_zip = False
        if zip_mode:
            out_as_zip = st.checkbox("Download result as ZIP (replace Datumaro JSON inside ZIP)", value=True, key="dm_zip_replace_2")

        if st.button("Merge ALL shapes into ONE track_id", type="primary"):
            merged, changed = datumaro_merge_all_shapes_to_one_track(
                deepcopy(datumaro_data),
                label_name=label_choice2,
                target_track_id=int(target_track_id),
                force_add_track_id=force_add_track_id,
                set_group=set_group2,
                add_instance_id=add_instance_id2,
                reindex_annotation_ids=reindex_ids2,
            )
            st.success(f"Updated {changed} annotations: all '{label_choice2}' → track_id={int(target_track_id)} across the dataset.")

            out_json = json.dumps(merged, ensure_ascii=False, indent=2).encode("utf-8")

            if zip_mode and out_as_zip and tmpdir:
                dm_json_path, _ = _find_datumaro_json_in_dir(tmpdir)
                if dm_json_path is None:
                    st.warning("Could not re-find Datumaro JSON to replace; downloading JSON instead.")
                    st.download_button("Download merged Datumaro JSON", data=out_json, file_name="merged_datumaro.json", mime="application/json")
                else:
                    dm_json_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
                    out_zip = _rezip_folder_to_bytes(tmpdir)
                    st.download_button("Download merged Datumaro ZIP", data=out_zip, file_name="merged_datumaro.zip", mime="application/zip")
            else:
                st.download_button("Download merged Datumaro JSON", data=out_json, file_name="merged_datumaro.json", mime="application/json")


# ============================================================
# Cleanup zip temp dir
# ============================================================

if tmpdir is not None:
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass
