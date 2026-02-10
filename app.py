import streamlit as st
import xml.etree.ElementTree as ET
import json
import math
import zipfile
import tempfile
from io import BytesIO
from copy import deepcopy
from collections import defaultdict
from pathlib import Path

st.set_page_config(page_title="CVAT / Datumaro Track & Shape Merger", layout="wide")
st.title("CVAT / Datumaro Track & Shape Merger (2D + 3D)")

st.markdown(
    """
Upload one of:
- **CVAT XML (video track XML)** → merge tracks across segments/jobs, or merge ALL tracks of a label.
- **CVAT XML (image/shapes XML)** → convert per-image shapes into ONE track across the whole task.
- **Datumaro JSON** or **Datumaro ZIP** (2D or 3D including `cuboid_3d`) → merge by label + track identity (rewrite `track_id`, unify `group`, add `instance_id`, optional reindex).

**Tip:** A Datumaro annotation `"id"` is a unique record id, not the track identity. Track identity is `attributes.track_id` (and commonly `group`).
"""
)

uploaded = st.file_uploader("Upload .xml / .json / .zip", type=["xml", "json", "zip"])

# ============================================================
# Shared helpers
# ============================================================

def _try_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _next_track_id_xml(root: ET.Element) -> int:
    ids = []
    for t in root.findall("track"):
        v = _try_int(t.get("id", None))
        if v is not None:
            ids.append(v)
    return (max(ids) + 1) if ids else 0

def _export_xml_bytes(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)

# ============================================================
# CVAT VIDEO XML (track-based) merger helpers
# ============================================================

def parse_segments_xml(root):
    segments = []
    meta = root.find("meta")
    if meta is None:
        return segments

    # Multi-segment task
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

    # Single job export fallback
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
            fr = _try_int(node.get("frame", None))
            if fr is not None:
                frames.append(fr)
    if not frames:
        return [{"id": "0", "start": 0, "stop": 0, "source": "fallback"}]
    return [{"id": "0", "start": min(frames), "stop": max(frames), "source": "fallback"}]

def collect_job_tracks_xml(root, segments):
    """
    Map which track IDs appear in each segment's frame range (based on <box frame=...> etc.)
    """
    job_tracks = {seg["id"]: {} for seg in segments}

    for track in root.findall("track"):
        tid = track.get("id")
        label = track.get("label", "")

        attrs = {}
        for a in track.findall("attribute"):
            name = a.get("name")
            if name:
                attrs[name] = (a.text or "").strip()

        # scan any geometry nodes under track: box/polygon/polyline/points/cuboid_3d etc.
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
    """
    Only for <box> nodes: sorted list (frame, cx, cy)
    """
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
    """
    For continuity suggestion, we use box center continuity when possible.
    """
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
                # If no boxes, still provide a fallback length based on any geometry nodes
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
    """
    Suggest exactly ONE track per segment.
    If box geometry exists: use center continuity; otherwise fall back to longest in segment.
    """
    suggestion = {}
    prev_tid = None
    prev_sid = None

    ordered_segments = sorted(segments, key=lambda s: s["start"])
    for seg in ordered_segments:
        sid = seg["id"]
        candidates = track_info.get(sid, {})
        if not candidates:
            continue

        if len(candidates) == 1:
            tid = next(iter(candidates.keys()))
            suggestion[sid] = [tid]
            prev_tid, prev_sid = tid, sid
            continue

        # If no history: choose longest
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
                # no geometry for distance: penalize slightly, prefer others
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
    """
    Build one merged <track> from selected track IDs per segment, copying geometry nodes whose frame is within segment.
    This merges into ONE track (same label as first selected).
    """
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
                frame = _try_int(node.get("frame", None), default=None)
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

    # attributes first
    for a in reversed(base_attrs):
        new_track.insert(0, a)

    # sort geometry nodes by frame (keep attributes at top)
    geom = [n for n in list(new_track) if n.tag != "attribute"]
    geom_sorted = sorted(geom, key=lambda n: int(n.get("frame", 0)))
    for n in geom:
        new_track.remove(n)
    for n in geom_sorted:
        new_track.append(n)

    # remove originals used
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

def merge_all_tracks_of_label_into_one_track_xml(root, label_name: str, resolve_same_frame="largest"):
    """
    For track-based XML: merge ALL tracks of a label into ONE track.
    If multiple same-tag nodes exist at same frame, pick one (largest for box, first otherwise).
    """
    tracks = [t for t in root.findall("track") if t.get("label", "") == label_name]
    if not tracks:
        return None, 0

    new_id = _next_track_id_xml(root)
    new_track = ET.Element("track", {"id": str(new_id), "label": label_name, "source": "manual", "group": "0"})

    # copy attributes from first track
    for a in tracks[0].findall("attribute"):
        new_track.append(deepcopy(a))

    # gather candidates per frame per tag
    per_frame_tag = defaultdict(lambda: defaultdict(list))
    for t in tracks:
        for node in list(t):
            if node.tag == "attribute":
                continue
            fr = _try_int(node.get("frame", None))
            if fr is None:
                continue
            per_frame_tag[fr][node.tag].append(deepcopy(node))

    chosen_nodes = []
    for fr, tags in per_frame_tag.items():
        for tag, nodes in tags.items():
            if len(nodes) == 1:
                chosen_nodes.append(nodes[0])
            else:
                if tag == "box" and resolve_same_frame == "largest":
                    chosen_nodes.append(max(nodes, key=_box_area_node))
                else:
                    chosen_nodes.append(nodes[0])

    chosen_nodes.sort(key=lambda n: int(n.get("frame", 0)))
    for n in chosen_nodes:
        new_track.append(n)

    for t in tracks:
        root.remove(t)
    root.append(new_track)
    return root, len(tracks)

# ============================================================
# CVAT IMAGE/SHAPES XML -> ONE TRACK conversion
# ============================================================

def _shape_area_for_collision(node):
    if node.tag == "box":
        return _box_area_node(node)
    # for polygon/polyline/points: no easy area; return 0
    return 0.0

def shapes_to_single_track_xml(
    root: ET.Element,
    shape_tag: str,
    label_name: str,
    resolve_same_frame: str = "largest",  # largest/first
    remove_original_shapes: bool = True,
):
    """
    Convert <image><box|polygon|polyline|points label=...> shapes into ONE <track> with frame=image/@id.
    Works for tags: box, polygon, polyline, points
    """
    images = root.findall("image")
    if not images:
        return None, "No <image> elements found."

    # collect per frame shapes
    per_frame = defaultdict(list)
    for img in images:
        frame = _try_int(img.get("id", "0"), default=0)
        for node in img.findall(shape_tag):
            if node.get("label") == label_name:
                per_frame[frame].append(node)

    if not per_frame:
        return None, f"No <{shape_tag}> shapes found for label '{label_name}'."

    # choose one per frame
    chosen = {}
    for frame, nodes in per_frame.items():
        if len(nodes) == 1:
            chosen[frame] = nodes[0]
        else:
            if resolve_same_frame == "largest":
                chosen[frame] = max(nodes, key=_shape_area_for_collision)
            else:
                chosen[frame] = nodes[0]

    # create new track
    new_id = _next_track_id_xml(root)
    new_track = ET.Element("track", {"id": str(new_id), "label": label_name, "source": "manual", "group": "0"})

    # build track nodes
    for frame in sorted(chosen.keys()):
        src = chosen[frame]
        attrib = dict(src.attrib)

        # CVAT track-style: must have frame/outside/keyframe/occluded/z_order + geometry fields
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
            # these use "points" attribute
            if "points" not in attrib:
                continue
            base.update({"points": attrib["points"]})
            dst = ET.Element(shape_tag, base)

        else:
            return None, f"Unsupported shape tag: {shape_tag}"

        # copy <attribute> children if exist
        for a in src.findall("attribute"):
            dst.append(deepcopy(a))

        new_track.append(dst)

    # remove originals
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

def is_datumaro_json(data: dict) -> bool:
    return isinstance(data, dict) and "items" in data and "categories" in data and "label" in data["categories"]

def datumaro_label_map(data: dict):
    labels = data["categories"]["label"].get("labels", [])
    return {i: lbl.get("name", f"label_{i}") for i, lbl in enumerate(labels)}

def collect_datumaro_tracks(data: dict):
    """
    tracks[label_name][track_id] = {"count": int, "min_frame": int|None, "max_frame": int|None, "types": set[str], "with_track_id": int}
    """
    label_map = datumaro_label_map(data)
    tracks = defaultdict(lambda: defaultdict(lambda: {"count": 0, "min_frame": None, "max_frame": None, "types": set(), "with_track_id": 0}))

    for item in data.get("items", []):
        frame = item.get("attr", {}).get("frame", None)
        for ann in item.get("annotations", []):
            label_id = ann.get("label_id", -1)
            label_name = label_map.get(label_id, f"label_{label_id}")
            atype = ann.get("type", "unknown")
            attrs = ann.get("attributes", {}) or {}

            if "track_id" not in attrs:
                continue

            tid = attrs["track_id"]
            rec = tracks[label_name][tid]
            rec["count"] += 1
            rec["types"].add(atype)
            rec["with_track_id"] += 1

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
    """
    Rewrite ALL annotations of a label to the same target_track_id.
    If an annotation lacks track_id and force_add_track_id=True, it will be added.
    """
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

def _find_datumaro_json_in_dir(root_dir: Path):
    for jf in sorted(root_dir.rglob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if is_datumaro_json(data):
            return jf, data
    return None, None

def load_datumaro_from_zip_bytes(zip_bytes: bytes):
    """
    Extract zip -> find a Datumaro JSON -> return (tmpdir_path, json_path, data)
    Caller must clean up tmpdir.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="datumaro_zip_"))
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
        z.extractall(tmpdir)
    json_path, data = _find_datumaro_json_in_dir(tmpdir)
    if json_path is None:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None, None, None
    return tmpdir, json_path, data

def rezip_folder_to_bytes(folder: Path) -> bytes:
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_dir():
                continue
            z.write(p, arcname=str(p.relative_to(folder)))
    bio.seek(0)
    return bio.getvalue()

# ============================================================
# Main flow
# ============================================================

if uploaded is None:
    st.stop()

fname = uploaded.name.lower()

# ----------------------------------------------------------------
# XML MODE
# ----------------------------------------------------------------
if fname.endswith(".xml"):
    st.header("Mode: CVAT XML")

    try:
        tree = ET.parse(uploaded)
        root = tree.getroot()
    except ET.ParseError as e:
        st.error(f"Failed to parse XML: {e}")
        st.stop()

    has_images = len(root.findall("image")) > 0
    has_tracks = len(root.findall("track")) > 0

    tabs = st.tabs(["Auto Detect", "Video Track XML", "Image/Shapes XML → Track"])

    # ---------------- AUTO DETECT
    with tabs[0]:
        if has_images and not has_tracks:
            st.info("Detected **image/shapes-style** CVAT XML (has <image>, no <track>). Use Shapes→Track tab.")
        elif has_tracks:
            st.info("Detected **track-style** CVAT XML (has <track>). Use Video Track XML tab.")
        else:
            st.warning("Could not clearly detect. Try the tabs manually.")

    # ---------------- VIDEO TRACK XML
    with tabs[1]:
        if not has_tracks:
            st.warning("This XML does not contain <track> elements.")
        else:
            st.subheader("A) Merge selected tracks across segments/jobs into ONE track")
            segments = parse_segments_xml(root)
            if not segments:
                segments = fallback_single_segment_from_track_xml(root)
                st.info("No jobs/segments found — using a single fallback segment spanning all frames.")

            job_tracks = collect_job_tracks_xml(root, segments)
            track_info = build_track_info_xml(root, segments, job_tracks)
            auto_suggestion = suggest_track_chain_xml(segments, track_info)

            st.caption("Detected segments/jobs:")
            for seg in sorted(segments, key=lambda s: s["start"]):
                sid = seg["id"]
                tracks = job_tracks.get(sid, {})
                st.write(f"- Segment/Job {sid}: frames {seg['start']}–{seg['stop']} | tracks found: {len(tracks)}")

            selected_tracks = build_selection_ui_xml(segments, job_tracks, track_info, auto_suggestion)

            if st.button("Generate merged XML (ONE track from selections)", type="primary"):
                merged_root = merge_tracks_xml_by_segments(deepcopy(root), segments, selected_tracks)
                if merged_root is None:
                    st.error("No valid merged track built. Check selections.")
                else:
                    st.success("Merged track created from your selections.")
                    st.download_button(
                        "Download merged CVAT XML",
                        data=_export_xml_bytes(merged_root),
                        file_name="cvat_merged_track.xml",
                        mime="application/xml",
                    )

            st.markdown("---")
            st.subheader("B) Merge ALL tracks of a label into ONE track (track-style XML)")

            labels = sorted({t.get("label", "") for t in root.findall("track") if t.get("label", "")})
            if not labels:
                st.warning("No labels found in <track> elements.")
            else:
                col1, col2 = st.columns([2, 2])
                with col1:
                    merge_label = st.selectbox("Label", labels, key="xml_merge_all_label")
                with col2:
                    resolve = st.selectbox("If multiple shapes exist on the same frame, keep:", ["largest", "first"], key="xml_merge_all_resolve")

                if st.button("Merge ALL tracks of this label into ONE track", key="xml_merge_all_btn"):
                    merged_root, count = merge_all_tracks_of_label_into_one_track_xml(deepcopy(root), merge_label, resolve_same_frame=resolve)
                    if merged_root is None:
                        st.error("No tracks found for that label.")
                    else:
                        st.success(f"Merged {count} tracks of label '{merge_label}' into ONE track.")
                        st.download_button(
                            "Download merged CVAT XML",
                            data=_export_xml_bytes(merged_root),
                            file_name=f"cvat_merge_all_{merge_label}.xml",
                            mime="application/xml",
                        )

    # ---------------- SHAPES XML -> TRACK
    with tabs[2]:
        if not has_images:
            st.warning("This XML does not contain <image> elements (not shapes-style).")
        else:
            st.subheader("Convert per-image shapes into ONE track across the whole task")

            # discover available shape tags and labels
            shape_tags = ["box", "polygon", "polyline", "points"]
            available = []
            for tag in shape_tags:
                found = False
                for img in root.findall("image"):
                    if img.find(tag) is not None:
                        found = True
                        break
                if found:
                    available.append(tag)

            if not available:
                st.warning("No supported shapes found inside <image> (box/polygon/polyline/points).")
            else:
                shape_tag = st.selectbox("Shape type to convert", available, key="shape_tag_select")

                # labels for that shape type
                labels = sorted({
                    n.get("label", "")
                    for img in root.findall("image")
                    for n in img.findall(shape_tag)
                    if n.get("label", "")
                })
                if not labels:
                    st.warning(f"No labels found for <{shape_tag}> shapes.")
                else:
                    label_name = st.selectbox("Label to convert into a track", labels, key="shape_label_select")
                    resolve = st.selectbox("If multiple shapes exist on the same frame, keep:", ["largest", "first"], key="shape_resolve_select")
                    remove_orig = st.checkbox("Remove original shapes after creating track", value=True, key="shape_remove_orig")

                    if st.button("Convert shapes → ONE track", type="primary", key="shape_to_track_btn"):
                        merged_root, msg = shapes_to_single_track_xml(
                            deepcopy(root),
                            shape_tag=shape_tag,
                            label_name=label_name,
                            resolve_same_frame=resolve,
                            remove_original_shapes=remove_orig,
                        )
                        if merged_root is None:
                            st.error(msg)
                        else:
                            st.success(msg)
                            st.download_button(
                                "Download XML with new track",
                                data=_export_xml_bytes(merged_root),
                                file_name=f"cvat_shapes_to_track_{shape_tag}_{label_name}.xml",
                                mime="application/xml",
                            )

# ----------------------------------------------------------------
# DATUMARO JSON / ZIP MODE
# ----------------------------------------------------------------
else:
    st.header("Mode: Datumaro (2D/3D)")

    # Load data either from JSON or ZIP
    zip_context = None  # (tmpdir, json_path) if zip
    data = None

    if fname.endswith(".json"):
        try:
            data = json.load(uploaded)
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")
            st.stop()

        if not is_datumaro_json(data):
            st.error("This JSON does not look like Datumaro (missing items/categories/label).")
            st.stop()

    elif fname.endswith(".zip"):
        raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        tmpdir, json_path, data = load_datumaro_from_zip_bytes(raw)
        if data is None:
            st.error("Could not find a Datumaro JSON inside the ZIP.")
            st.stop()
        zip_context = (tmpdir, json_path)
        st.caption(f"Loaded Datumaro JSON from ZIP path: {json_path.relative_to(tmpdir)}")

    else:
        st.error("Unsupported file type. Upload .xml / .json / .zip")
        st.stop()

    label_map = datumaro_label_map(data)
    # gather label names from category list (even if no tracks exist yet)
    all_label_names = sorted({nm for nm in label_map.values()})

    if not all_label_names:
        st.warning("No labels found in datumaro categories.")
        st.stop()

    tracks = collect_datumaro_tracks(data)

    tabs = st.tabs(["Merge selected track_ids", "Merge ALL shapes into ONE track_id (whole task)"])

    # ---------------- Merge selected track_ids
    with tabs[0]:
        st.subheader("Merge selected track_ids within a label (track identity rewrite)")

        label_choice = st.selectbox("Label/class", all_label_names, key="dm_label_sel_1")

        # Build display strings: Label | track_id | frames | count | types
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
            st.warning("No track_id found for this label. Use the 'Merge ALL shapes' tab if you want to create one track for all shapes.")
        else:
            selected_disps = st.multiselect(
                "Select track_ids to merge (shown with label name)",
                tid_options,
                default=tid_options[: min(3, len(tid_options))],
                key="dm_tid_multi",
            )
            selected_tids = [tid_lookup[d] for d in selected_disps]

            colA, colB, colC = st.columns([2, 2, 2])
            with colA:
                target_mode = st.radio("Target track_id", ["Use smallest selected", "Enter manually"], horizontal=True, key="dm_target_mode")
            with colB:
                set_group = st.checkbox("Unify 'group' to target track_id", value=True, key="dm_set_group")
            with colC:
                add_instance_id = st.checkbox("Add attributes.instance_id = target track_id", value=True, key="dm_instance_id")

            target_tid = None
            if target_mode == "Enter manually":
                target_tid = st.number_input("Target track_id (int)", min_value=0, value=int(min(selected_tids)) if selected_tids else 0, step=1, key="dm_target_tid_input")

            reindex_ids = st.checkbox("Reindex annotation 'id' fields (keep unique, 0..N-1)", value=False, key="dm_reindex")

            if st.button("Merge selected track_ids", type="primary", key="dm_merge_selected_btn"):
                if len(selected_tids) < 2:
                    st.error("Select at least 2 track_ids to merge.")
                else:
                    tgt = int(min(selected_tids)) if target_mode == "Use smallest selected" else int(target_tid)
                    merged, changed = datumaro_merge_selected_track_ids(
                        deepcopy(data),
                        label_name=label_choice,
                        source_track_ids=selected_tids,
                        target_track_id=tgt,
                        set_group=set_group,
                        add_instance_id=add_instance_id,
                        reindex_annotation_ids=reindex_ids,
                    )
                    st.success(f"Updated {changed} annotations: merged {len(selected_tids)} track_ids → {tgt} for label '{label_choice}'.")

                    out_json = json.dumps(merged, ensure_ascii=False, indent=2).encode("utf-8")

                    if zip_context:
                        tmpdir, json_path = zip_context
                        # write back and rezip
                        json_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
                        out_zip = rezip_folder_to_bytes(tmpdir)
                        st.download_button(
                            "Download merged Datumaro ZIP",
                            data=out_zip,
                            file_name="merged_datumaro.zip",
                            mime="application/zip",
                        )
                    else:
                        st.download_button(
                            "Download merged Datumaro JSON",
                            data=out_json,
                            file_name="merged_datumaro.json",
                            mime="application/json",
                        )

    # ---------------- Merge ALL shapes into ONE track_id
    with tabs[1]:
        st.subheader("Merge ALL shapes of a label into ONE track_id across the whole task")

        label_choice2 = st.selectbox("Label/class", all_label_names, key="dm_label_sel_2")

        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            target_track_id = st.number_input("Target track_id", min_value=0, value=0, step=1, key="dm_merge_all_target")
        with col2:
            force_add_track_id = st.checkbox("If an annotation lacks track_id, add it", value=True, key="dm_force_add")
        with col3:
            set_group2 = st.checkbox("Unify 'group' to target track_id", value=True, key="dm_set_group2")

        col4, col5 = st.columns([2, 2])
        with col4:
            add_instance_id2 = st.checkbox("Add attributes.instance_id = target track_id", value=True, key="dm_instance_id2")
        with col5:
            reindex_ids2 = st.checkbox("Reindex annotation 'id' fields (keep unique, 0..N-1)", value=False, key="dm_reindex2")

        if st.button("Merge ALL shapes into ONE track_id", type="primary", key="dm_merge_all_btn"):
            merged, changed = datumaro_merge_all_shapes_to_one_track(
                deepcopy(data),
                label_name=label_choice2,
                target_track_id=int(target_track_id),
                force_add_track_id=force_add_track_id,
                set_group=set_group2,
                add_instance_id=add_instance_id2,
                reindex_annotation_ids=reindex_ids2,
            )
            st.success(f"Updated {changed} annotations: all '{label_choice2}' → track_id={int(target_track_id)} across the dataset.")

            out_json = json.dumps(merged, ensure_ascii=False, indent=2).encode("utf-8")

            if zip_context:
                tmpdir, json_path = zip_context
                json_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
                out_zip = rezip_folder_to_bytes(tmpdir)
                st.download_button(
                    "Download merged Datumaro ZIP",
                    data=out_zip,
                    file_name="merged_datumaro.zip",
                    mime="application/zip",
                )
            else:
                st.download_button(
                    "Download merged Datumaro JSON",
                    data=out_json,
                    file_name="merged_datumaro.json",
                    mime="application/json",
                )

    # clean up tmpdir if zip
    if zip_context:
        tmpdir, _ = zip_context
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
