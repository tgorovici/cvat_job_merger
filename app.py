import streamlit as st
import xml.etree.ElementTree as ET
from io import BytesIO
from copy import deepcopy
import math
import json
import zipfile
from typing import Dict, List, Tuple, Any, Optional

st.set_page_config(page_title="CVAT / Datumaro Track & Shape Merger", layout="wide")

st.title("CVAT / Datumaro Track & Shape Merger (2D + 3D)")
st.caption(
    "Supports: CVAT Video XML, CVAT XML ZIP, Datumaro JSON (2D + 3D). "
    "Merge tracks by label, convert shapes to tracks, export back."
)

# -----------------------------
# Format lists (UI only)
# -----------------------------
COMMON_EXPORT_FORMATS = [
    # CVAT
    "CVAT 1.1 (Video/XML)",
    "CVAT 1.1 (Images/XML)",
    "CVAT for video 1.1",
    "CVAT for images 1.1",
    "CVAT (Datumaro)",
    "CVAT (COCO)",
    # Datumaro
    "Datumaro (JSON)",
    "Datumaro (COCO)",
    "Datumaro (YOLO)",
    "Datumaro (VOC)",
    # 3D-ish common labels (mostly for user clarity)
    "CVAT 3D (Datumaro JSON)",
    "CVAT 3D (PCD/Cuboid via Datumaro JSON)",
]

# -----------------------------
# Helpers: misc
# -----------------------------
def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def bytes_download(data_bytes: bytes) -> BytesIO:
    return BytesIO(data_bytes)

def next_available_int(existing: List[int], start_at: int = 0) -> int:
    s = set(existing)
    v = start_at
    while v in s:
        v += 1
    return v

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# -----------------------------
# Input loader: XML or ZIP(XML) or JSON(Datumaro)
# -----------------------------
def load_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith(".zip"):
        # find first .xml inside
        with zipfile.ZipFile(BytesIO(raw), "r") as zf:
            xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
            if not xml_names:
                raise ValueError("ZIP has no .xml files inside.")
            # prefer annotations.xml if exists
            xml_name = None
            for cand in xml_names:
                if cand.lower().endswith("annotations.xml"):
                    xml_name = cand
                    break
            if xml_name is None:
                xml_name = xml_names[0]
            xml_bytes = zf.read(xml_name)
            tree = ET.ElementTree(ET.fromstring(xml_bytes))
            return ("cvat_xml", tree)

    if name.endswith(".xml"):
        tree = ET.parse(BytesIO(raw))
        return ("cvat_xml", tree)

    if name.endswith(".json"):
        obj = json.loads(raw.decode("utf-8"))
        return ("datumaro_json", obj)

    raise ValueError("Unsupported file type. Use .xml, .zip (with xml), or .json")

# -----------------------------
# CVAT XML parsing / operations
# -----------------------------
def xml_parse_segments(root):
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

def xml_collect_job_tracks(root, segments):
    job_tracks = {seg["id"]: {} for seg in segments}
    for track in root.findall("track"):
        tid = track.get("id")
        label = track.get("label", "")

        attrs = {}
        for a in track.findall("attribute"):
            name = a.get("name")
            if name:
                attrs[name] = (a.text or "").strip()

        for child in list(track):
            # boxes/polygons/etc inside track have frame attribute
            if child.tag in ("box", "polygon", "polyline", "points", "cuboid", "mask", "skeleton"):
                frame = safe_int(child.get("frame"), None)
                if frame is None:
                    continue
                for seg in segments:
                    if seg["start"] <= frame <= seg["stop"]:
                        seg_tracks = job_tracks.setdefault(seg["id"], {})
                        info = seg_tracks.setdefault(
                            tid,
                            {"label": label, "attributes": attrs, "frames": []},
                        )
                        info["frames"].append(frame)
    return job_tracks

def xml_get_track_centers(track):
    # Only for "box" tracks (2D). If not box, we fallback to "first frame" only scoring.
    boxes = []
    for box in track.findall("box"):
        frame = safe_int(box.get("frame"), None)
        if frame is None:
            continue
        try:
            xtl = float(box.get("xtl", 0))
            ytl = float(box.get("ytl", 0))
            xbr = float(box.get("xbr", 0))
            ybr = float(box.get("ybr", 0))
        except Exception:
            continue
        cx = (xtl + xbr) / 2.0
        cy = (ytl + ybr) / 2.0
        boxes.append((frame, cx, cy))
    boxes.sort(key=lambda x: x[0])
    return boxes

def xml_build_track_info(root, segments, job_tracks):
    track_map = {t.get("id"): t for t in root.findall("track")}
    info = {seg["id"]: {} for seg in segments}

    for seg in segments:
        sid = seg["id"]
        for tid, meta in job_tracks.get(sid, {}).items():
            t = track_map.get(tid)
            if t is None:
                continue

            # geometry scoring (only for boxes)
            centers = xml_get_track_centers(t)
            frames = []
            for child in list(t):
                if child.tag in ("box", "polygon", "polyline", "points", "cuboid", "mask", "skeleton"):
                    f = safe_int(child.get("frame"), None)
                    if f is not None:
                        frames.append(f)
            if not frames:
                continue

            frames.sort()
            entry = {
                "label": meta["label"],
                "attributes": meta["attributes"],
                "start_frame": frames[0],
                "end_frame": frames[-1],
                "has_box_centers": bool(centers),
                "first_center": (centers[0][1], centers[0][2]) if centers else None,
                "last_center": (centers[-1][1], centers[-1][2]) if centers else None,
            }
            info[sid][tid] = entry
    return info

def attr_signature(attrs: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(attrs.items()))

def xml_suggest_track_chain(segments, track_info):
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

        if prev_tid is None or prev_sid is None:
            best_tid = max(
                candidates.items(),
                key=lambda kv: kv[1]["end_frame"] - kv[1]["start_frame"]
            )[0]
            suggestion[sid] = [best_tid]
            prev_tid, prev_sid = best_tid, sid
            continue

        prev_info = track_info.get(prev_sid, {}).get(prev_tid)
        if not prev_info:
            best_tid = max(
                candidates.items(),
                key=lambda kv: kv[1]["end_frame"] - kv[1]["start_frame"]
            )[0]
            suggestion[sid] = [best_tid]
            prev_tid, prev_sid = best_tid, sid
            continue

        prev_label = prev_info["label"]
        prev_attr_sig = attr_signature(prev_info["attributes"])

        best_tid = None
        best_score = float("inf")

        for tid, ci in candidates.items():
            score = 0.0
            if ci["label"] != prev_label:
                score += 10_000
            if attr_signature(ci["attributes"]) != prev_attr_sig:
                score += 1_000

            # center-distance scoring if both sides have box centers, otherwise just light penalty
            if prev_info["has_box_centers"] and ci["has_box_centers"]:
                score += euclidean(prev_info["last_center"], ci["first_center"])
            else:
                score += 10.0

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

def xml_find_root_shape_elements(root) -> List[ET.Element]:
    """
    In CVAT XML, non-track shapes might appear directly under root (besides <track> and <meta>).
    We'll consider typical annotation tags with a 'frame' attribute.
    """
    shapes = []
    for child in list(root):
        if child.tag in ("meta", "track"):
            continue
        if child.tag in ("box", "polygon", "polyline", "points", "cuboid", "mask", "skeleton"):
            if child.get("frame") is not None:
                shapes.append(child)
    return shapes

def xml_shape_to_track_transform(root, merge_labels: Optional[List[str]] = None) -> int:
    """
    Convert root-level shapes (no track) into tracks.
    Strategy: create ONE track per label, across entire task, and move those shapes into it.
    Returns number of shapes moved.
    """
    shapes = xml_find_root_shape_elements(root)
    if not shapes:
        return 0

    # existing track ids
    existing_ids = []
    for t in root.findall("track"):
        vi = safe_int(t.get("id"), None)
        if vi is not None:
            existing_ids.append(vi)

    # group shapes by label
    by_label: Dict[str, List[ET.Element]] = {}
    for sh in shapes:
        label = sh.get("label", "")
        if merge_labels and label not in merge_labels:
            continue
        by_label.setdefault(label, []).append(sh)

    moved = 0
    for label, elems in by_label.items():
        new_id = next_available_int(existing_ids, start_at=0)
        existing_ids.append(new_id)

        tr = ET.Element("track", {"id": str(new_id), "label": label, "source": "manual", "group": "0"})
        # move deepcopies, then remove originals
        # Ensure track children have outside/keyframe defaults
        copies = []
        for e in elems:
            c = deepcopy(e)
            if c.get("outside") is None:
                c.set("outside", "0")
            if c.get("keyframe") is None:
                c.set("keyframe", "1")
            copies.append(c)

        copies.sort(key=lambda x: safe_int(x.get("frame"), 0) or 0)
        for c in copies:
            tr.append(c)

        # remove original shapes from root
        for e in elems:
            if e in root:
                root.remove(e)
                moved += 1

        root.append(tr)

    return moved

def xml_merge_selected_tracks_into_one(root, segments, selected_tracks):
    """
    Build one merged <track> from selected tracks across segments.
    """
    existing_ids = []
    for t in root.findall("track"):
        vi = safe_int(t.get("id"), None)
        if vi is not None:
            existing_ids.append(vi)
    new_id = next_available_int(existing_ids, start_at=(max(existing_ids) + 1) if existing_ids else 0)

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

            for ann in list(orig):
                if ann.tag not in ("box", "polygon", "polyline", "points", "cuboid", "mask", "skeleton"):
                    continue
                frame = safe_int(ann.get("frame"), None)
                if frame is None:
                    continue
                if seg["start"] <= frame <= seg["stop"]:
                    new_track.append(deepcopy(ann))

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

    anns = [a for a in list(new_track) if a.tag in ("box", "polygon", "polyline", "points", "cuboid", "mask", "skeleton")]
    anns_sorted = sorted(anns, key=lambda b: safe_int(b.get("frame"), 0) or 0)
    for a in anns:
        new_track.remove(a)
    for a in anns_sorted:
        new_track.append(a)

    for tid in used_track_ids:
        t = track_map.get(tid)
        if t is not None and t in root:
            root.remove(t)

    root.append(new_track)
    return root

def xml_merge_tracks_by_label(root, label: str):
    """
    Merge ALL tracks of a given label into ONE track (concatenate their child ann elements by frame).
    """
    tracks = [t for t in root.findall("track") if t.get("label", "") == label]
    if len(tracks) <= 1:
        return False

    existing_ids = []
    for t in root.findall("track"):
        vi = safe_int(t.get("id"), None)
        if vi is not None:
            existing_ids.append(vi)
    new_id = next_available_int(existing_ids, start_at=(max(existing_ids) + 1) if existing_ids else 0)

    base = tracks[0]
    new_track = ET.Element("track", {
        "id": str(new_id),
        "label": label,
        "group": base.get("group", "0"),
        "source": base.get("source", "manual"),
    })

    # preserve first track attributes only (common in CVAT)
    for a in base.findall("attribute"):
        new_track.append(deepcopy(a))

    all_anns = []
    for t in tracks:
        for ann in list(t):
            if ann.tag in ("box", "polygon", "polyline", "points", "cuboid", "mask", "skeleton"):
                c = deepcopy(ann)
                if c.get("outside") is None:
                    c.set("outside", "0")
                if c.get("keyframe") is None:
                    c.set("keyframe", "1")
                all_anns.append(c)

    all_anns.sort(key=lambda b: safe_int(b.get("frame"), 0) or 0)
    for a in all_anns:
        new_track.append(a)

    # remove old tracks
    for t in tracks:
        if t in root:
            root.remove(t)
    root.append(new_track)
    return True

def xml_export(root):
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return bytes_download(xml_bytes)

# -----------------------------
# Datumaro JSON operations
# -----------------------------
def dm_get_label_map(dm: dict) -> Dict[int, str]:
    """
    Datumaro JSON typically has categories -> label -> labels.
    We'll try best-effort.
    """
    cat = dm.get("categories", {})
    label_cat = cat.get("label", {})
    labels = label_cat.get("labels", [])
    out = {}
    for i, item in enumerate(labels):
        # item can be string or dict {"name": "..."}
        if isinstance(item, str):
            out[i] = item
        elif isinstance(item, dict):
            out[i] = item.get("name", str(i))
        else:
            out[i] = str(i)
    return out

def dm_iter_items(dm: dict) -> List[dict]:
    # Datumaro JSON can be {"items":[...]} or sometimes direct list. We'll handle common.
    if isinstance(dm, dict) and "items" in dm and isinstance(dm["items"], list):
        return dm["items"]
    if isinstance(dm, list):
        return dm
    # fallback: if it looks like item list
    if isinstance(dm, dict) and all(k in dm for k in ("id", "annotations")):
        return [dm]
    raise ValueError("Unrecognized Datumaro JSON structure: expected {items:[...]}")

def dm_collect_existing_track_ids(dm_items: List[dict]) -> List[int]:
    ids = []
    for it in dm_items:
        for ann in it.get("annotations", []):
            attrs = ann.get("attributes", {}) or {}
            if "track_id" in attrs:
                try:
                    ids.append(int(attrs["track_id"]))
                except Exception:
                    pass
    return ids

def dm_collect_existing_annotation_ids(dm_items: List[dict]) -> List[int]:
    ids = []
    for it in dm_items:
        for ann in it.get("annotations", []):
            if "id" in ann:
                try:
                    ids.append(int(ann["id"]))
                except Exception:
                    pass
    return ids

def dm_is_tracked(ann: dict) -> bool:
    attrs = ann.get("attributes", {}) or {}
    return "track_id" in attrs

def dm_convert_shapes_to_tracks(dm: dict, labels_to_convert: Optional[List[str]] = None) -> Tuple[int, int]:
    """
    For annotations WITHOUT track_id -> assign new track_id per label (one track per label),
    set keyframe=true, and optionally unify annotation id per new track.
    Returns: (num_shapes_converted, num_tracks_created)
    """
    items = dm_iter_items(dm)
    label_map = dm_get_label_map(dm)

    existing_track_ids = dm_collect_existing_track_ids(items)
    next_tid = next_available_int(existing_track_ids, start_at=0)

    existing_ann_ids = dm_collect_existing_annotation_ids(items)
    next_ann_id = next_available_int(existing_ann_ids, start_at=(max(existing_ann_ids) + 1) if existing_ann_ids else 1)

    # We'll create one new track per label_name, if it has shapes without track_id
    label_to_new_tid: Dict[str, int] = {}
    label_to_new_ann_id: Dict[str, int] = {}

    converted = 0
    for it in items:
        anns = it.get("annotations", [])
        for ann in anns:
            if dm_is_tracked(ann):
                continue

            label_id = ann.get("label_id", None)
            label_name = label_map.get(label_id, f"label_id={label_id}")

            if labels_to_convert and label_name not in labels_to_convert:
                continue

            if label_name not in label_to_new_tid:
                label_to_new_tid[label_name] = next_tid
                next_tid += 1
                label_to_new_ann_id[label_name] = next_ann_id
                next_ann_id += 1

            ann.setdefault("attributes", {})
            ann["attributes"]["track_id"] = label_to_new_tid[label_name]
            ann["attributes"]["keyframe"] = True

            # unify Datumaro "id" to be same across the entire new track
            ann["id"] = label_to_new_ann_id[label_name]
            converted += 1

    return converted, len(label_to_new_tid)

def dm_merge_tracks_by_label(dm: dict, labels_to_merge: List[str]) -> Dict[str, dict]:
    """
    Merge ALL tracks (by track_id) that share the same label into a single track per label:
      - rewrite attributes.track_id to a single chosen/new id
      - rewrite annotation 'id' to a single id as well (so final Datumaro is clean)
    """
    items = dm_iter_items(dm)
    label_map = dm_get_label_map(dm)

    existing_track_ids = dm_collect_existing_track_ids(items)
    existing_ann_ids = dm_collect_existing_annotation_ids(items)

    next_tid = next_available_int(existing_track_ids, start_at=(max(existing_track_ids) + 1) if existing_track_ids else 0)
    next_ann_id = next_available_int(existing_ann_ids, start_at=(max(existing_ann_ids) + 1) if existing_ann_ids else 1)

    # Find which track_ids exist per label
    label_to_trackids = {ln: set() for ln in labels_to_merge}
    for it in items:
        for ann in it.get("annotations", []):
            attrs = ann.get("attributes", {}) or {}
            if "track_id" not in attrs:
                continue
            try:
                tid = int(attrs["track_id"])
            except Exception:
                continue
            ln = label_map.get(ann.get("label_id"), f"label_id={ann.get('label_id')}")
            if ln in label_to_trackids:
                label_to_trackids[ln].add(tid)

    # For each label: if multiple track ids -> merge into one new tid (or keep if only one)
    merge_plan: Dict[str, dict] = {}
    for ln, tids in label_to_trackids.items():
        tids = sorted(list(tids))
        if not tids:
            continue
        if len(tids) == 1:
            # still can unify id if you want, but typically unnecessary
            merge_plan[ln] = {"from": tids, "to_track_id": tids[0], "to_ann_id": None, "changed": False}
        else:
            merge_plan[ln] = {"from": tids, "to_track_id": next_tid, "to_ann_id": next_ann_id, "changed": True}
            next_tid += 1
            next_ann_id += 1

    # Apply
    changed_count = 0
    for it in items:
        for ann in it.get("annotations", []):
            attrs = ann.get("attributes", {}) or {}
            if "track_id" not in attrs:
                continue
            ln = label_map.get(ann.get("label_id"), f"label_id={ann.get('label_id')}")
            if ln not in merge_plan:
                continue
            plan = merge_plan[ln]
            if not plan["changed"]:
                continue
            try:
                tid = int(attrs["track_id"])
            except Exception:
                continue
            if tid in plan["from"]:
                ann.setdefault("attributes", {})
                ann["attributes"]["track_id"] = int(plan["to_track_id"])
                ann["attributes"]["keyframe"] = bool(ann["attributes"].get("keyframe", True))
                # unify Datumaro annotation "id" too
                ann["id"] = int(plan["to_ann_id"])
                changed_count += 1

    return {"plan": merge_plan, "changed_annotations": changed_count}

def dm_export(dm: dict) -> BytesIO:
    data = json.dumps(dm, ensure_ascii=False, indent=2).encode("utf-8")
    return bytes_download(data)

# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader("Upload: CVAT XML / CVAT XML ZIP / Datumaro JSON", type=["xml", "zip", "json"])

colA, colB = st.columns([1, 1])
with colA:
    st.subheader("Output format (UI)")
    desired_format = st.selectbox(
        "Choose desired export format (UI only; app exports same family as input unless you add converters).",
        COMMON_EXPORT_FORMATS,
        index=0
    )
with colB:
    st.subheader("Actions")
    st.write("Pick one or more actions, then click **Run**.")

action_merge_tracks_by_label = st.checkbox("Merge tracks by label/class (collapse all tracks of same label into 1)")
action_convert_shapes_to_tracks = st.checkbox("Convert shapes (no track_id) into tracks (next available track_id)")
action_xml_merge_segment_chain = st.checkbox("CVAT XML: merge across jobs/segments (suggest chain + override)")

st.divider()

if uploaded_file is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    input_kind, payload = load_uploaded_file(uploaded_file)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

# -----------------------------
# CVAT XML flow
# -----------------------------
if input_kind == "cvat_xml":
    tree: ET.ElementTree = payload
    root = tree.getroot()

    st.subheader("Detected input: CVAT XML")
    segments = xml_parse_segments(root)
    has_segments = bool(segments)

    # Determine labels present in tracks and/or shapes
    track_labels = sorted({t.get("label", "") for t in root.findall("track") if t.get("label")})
    shape_labels = sorted({s.get("label", "") for s in xml_find_root_shape_elements(root) if s.get("label")})
    all_labels = sorted(set(track_labels) | set(shape_labels))

    left, right = st.columns([1, 1])
    with left:
        st.write("**Labels found**")
        st.write(", ".join(all_labels) if all_labels else "(none)")
    with right:
        st.write("**Segments/jobs found**")
        if has_segments:
            st.write(f"{len(segments)} segments/jobs in <meta>")
        else:
            st.write("No segments/jobs in <meta> (single-task / no split)")

    st.divider()

    # Label selection for merge
    st.subheader("Select labels/classes to apply merges on")
    labels_selected = st.multiselect(
        "Choose label(s) (leave empty = all labels for the chosen actions)",
        options=all_labels,
        default=all_labels[:1] if all_labels else []
    )

    # If segment chain merge is enabled, show suggestion UI
    selected_tracks_by_segment = None
    if action_xml_merge_segment_chain and has_segments:
        job_tracks = xml_collect_job_tracks(root, segments)
        track_info = xml_build_track_info(root, segments, job_tracks)
        auto_suggestion = xml_suggest_track_chain(segments, track_info)

        st.subheader("Segments & track candidates (shows Label | id)")
        for seg in sorted(segments, key=lambda s: s["start"]):
            sid = seg["id"]
            tracks = job_tracks.get(sid, {})
            if tracks:
                track_ids = list(tracks.keys())
                pretty = []
                for tid in track_ids:
                    label = tracks[tid]["label"]
                    if labels_selected and label not in labels_selected:
                        continue
                    pretty.append(f"{label} | xml_track_id={tid}")
                st.write(f"Segment {sid}: frames {seg['start']}–{seg['stop']} | {len(pretty)} tracks: {', '.join(pretty) if pretty else '(filtered out)'}")
            else:
                st.write(f"Segment {sid}: frames {seg['start']}–{seg['stop']} | no tracks")

        st.markdown("#### Choose track(s) per segment (suggested chain marked)")
        selected_tracks_by_segment = {}
        for seg in sorted(segments, key=lambda s: s["start"]):
            sid = seg["id"]
            tracks = job_tracks.get(sid, {})
            if not tracks:
                continue

            options = []
            option_to_tid = {}
            default = []

            for tid, meta in tracks.items():
                label = meta["label"]
                if labels_selected and label not in labels_selected:
                    continue

                desc = f"{label} | xml_track_id={tid}"
                if auto_suggestion.get(sid) == [tid]:
                    desc += "  ⟵ suggested"
                    default = [desc]

                options.append(desc)
                option_to_tid[desc] = tid

            if not options:
                continue

            chosen = st.multiselect(
                f"Segment {sid} tracks to merge into ONE object-chain:",
                options,
                default=default or options[:1],
                key=f"xml_seg_{sid}",
            )
            selected_tracks_by_segment[sid] = [option_to_tid[c] for c in chosen]

    st.divider()

    if st.button("Run (CVAT XML)", type="primary"):
        work_root = deepcopy(root)

        # 1) Convert shapes -> track (root-level shapes)
        if action_convert_shapes_to_tracks:
            moved = xml_shape_to_track_transform(work_root, merge_labels=labels_selected or None)
            st.success(f"Converted {moved} root-level shapes into track(s).")

        # 2) Merge tracks by label
        if action_merge_tracks_by_label:
            labels_to_process = labels_selected or list(sorted({t.get("label", "") for t in work_root.findall("track") if t.get("label")}))
            merged_any = 0
            for lb in labels_to_process:
                if xml_merge_tracks_by_label(work_root, lb):
                    merged_any += 1
            st.success(f"Merged tracks by label for {merged_any} label(s).")

        # 3) Merge across segments into one track (selected chain)
        if action_xml_merge_segment_chain and has_segments and selected_tracks_by_segment:
            merged = xml_merge_selected_tracks_into_one(work_root, segments, selected_tracks_by_segment)
            if merged is None:
                st.warning("Segment-chain merge produced no merged track (check selections).")
            else:
                st.success("Merged selected tracks across segments into one track (and removed originals used).")

        out = xml_export(work_root)
        st.download_button(
            "Download merged XML",
            data=out,
            file_name="merged_annotations.xml",
            mime="application/xml",
        )

# -----------------------------
# Datumaro JSON flow
# -----------------------------
else:
    dm: dict = payload
    st.subheader("Detected input: Datumaro JSON")

    try:
        items = dm_iter_items(dm)
    except Exception as e:
        st.error(f"Unrecognized Datumaro JSON: {e}")
        st.stop()

    label_map = dm_get_label_map(dm)

    # Scan labels present
    present_label_ids = set()
    tracked_count = 0
    shape_count = 0
    for it in items:
        for ann in it.get("annotations", []):
            present_label_ids.add(ann.get("label_id"))
            if dm_is_tracked(ann):
                tracked_count += 1
            else:
                shape_count += 1

    label_names = sorted({label_map.get(i, f"label_id={i}") for i in present_label_ids})

    st.write(f"Items: **{len(items)}** | tracked annotations (have track_id): **{tracked_count}** | shapes (no track_id): **{shape_count}**")
    st.write("**Labels found:**", ", ".join(label_names) if label_names else "(none)")

    st.divider()

    labels_selected = st.multiselect(
        "Choose labels/classes to apply merges on (leave empty = all labels for chosen actions)",
        options=label_names,
        default=label_names[:1] if label_names else []
    )

    st.info(
        "Note: this app treats Datumaro annotations **without** attributes.track_id as 'shapes' "
        "and can convert them into tracks by assigning a **new** track_id. "
        "Your uploaded JSON contains both types (with and without track_id) in the same frames. "
        ":contentReference[oaicite:2]{index=2}"
    )

    st.divider()

    if st.button("Run (Datumaro JSON)", type="primary"):
        work = deepcopy(dm)

        # 1) Convert shapes -> tracks
        if action_convert_shapes_to_tracks:
            converted, created = dm_convert_shapes_to_tracks(work, labels_to_convert=labels_selected or None)
            st.success(f"Converted {converted} shape annotations into tracks. New tracks created: {created}.")

        # 2) Merge tracks by label (and unify both track_id + annotation id)
        if action_merge_tracks_by_label:
            labels_to_merge = labels_selected or label_names
            res = dm_merge_tracks_by_label(work, labels_to_merge=labels_to_merge)
            plan = res["plan"]
            changed = res["changed_annotations"]

            merged_labels = [ln for ln, p in plan.items() if p.get("changed")]
            if merged_labels:
                st.success(
                    f"Merged tracks for {len(merged_labels)} label(s). "
                    f"Rewrote track_id + annotation id for {changed} annotations."
                )
            else:
                st.info("No label had multiple track_ids to merge (nothing changed).")

        out = dm_export(work)
        st.download_button(
            "Download merged Datumaro JSON",
            data=out,
            file_name="merged_datumaro.json",
            mime="application/json",
        )
