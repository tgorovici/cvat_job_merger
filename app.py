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

st.set_page_config(page_title="CVAT / Datumaro Track Merger", layout="wide")
st.title("CVAT / Datumaro Track Merger (2D + 3D)")

st.markdown(
    """
Upload one of:
- **CVAT Video XML** (with or without segments/jobs) → merge selected tracks across segments into **one track**.
- **Datumaro JSON** (2D or 3D, including cuboid_3d) or **Datumaro ZIP** → merge tracks by **label + track_id remap**.

The UI always shows **Label name + track_id**, so you know what you’re merging.
"""
)

uploaded = st.file_uploader("Upload file", type=["xml", "json", "zip"])

# -----------------------------
# Common helpers
# -----------------------------
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# -----------------------------
# CVAT XML (video) helpers
# -----------------------------
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

def fallback_single_segment_from_xml(root):
    frames = []
    for tr in root.findall("track"):
        for box in tr.findall("box"):
            try:
                frames.append(int(box.get("frame", 0)))
            except:
                pass
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

        for box in track.findall("box"):
            frame = int(box.get("frame", 0))
            for seg in segments:
                if seg["start"] <= frame <= seg["stop"]:
                    seg_tracks = job_tracks.setdefault(seg["id"], {})
                    info = seg_tracks.setdefault(
                        tid,
                        {"label": label, "attributes": attrs, "frames": []},
                    )
                    info["frames"].append(frame)

    return job_tracks

def get_track_boxes_by_frame_xml(track):
    boxes = []
    for box in track.findall("box"):
        try:
            frame = int(box.get("frame", 0))
            xtl = float(box.get("xtl", 0))
            ytl = float(box.get("ytl", 0))
            xbr = float(box.get("xbr", 0))
            ybr = float(box.get("ybr", 0))
        except ValueError:
            continue
        cx = (xtl + xbr) / 2.0
        cy = (ytl + ybr) / 2.0
        boxes.append((frame, cx, cy))
    boxes.sort(key=lambda x: x[0])
    return boxes

def attr_signature(attrs):
    return tuple(sorted(attrs.items()))

def build_track_info_xml(root, segments, job_tracks):
    track_map = {t.get("id"): t for t in root.findall("track")}
    info = {seg["id"]: {} for seg in segments}

    for seg in segments:
        sid = seg["id"]
        for tid, meta in job_tracks.get(sid, {}).items():
            t = track_map.get(tid)
            if t is None:
                continue
            boxes = get_track_boxes_by_frame_xml(t)
            if not boxes:
                continue
            frames = [b[0] for b in boxes]
            info[sid][tid] = {
                "label": meta["label"],
                "attributes": meta["attributes"],
                "start_frame": min(frames),
                "end_frame": max(frames),
                "first_box": boxes[0],
                "last_box": boxes[-1],
            }
    return info

def suggest_track_chain_xml(segments, track_info):
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
        prev_center = (prev_info["last_box"][1], prev_info["last_box"][2])

        best_tid = None
        best_score = float("inf")

        for tid, ci in candidates.items():
            score = 0.0
            if ci["label"] != prev_label:
                score += 10_000
            if attr_signature(ci["attributes"]) != prev_attr_sig:
                score += 1_000
            cand_center = (ci["first_box"][1], ci["first_box"][2])
            score += euclidean(prev_center, cand_center)

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
    st.subheader("CVAT XML: Track selection per segment/job")
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
            f"Choose track(s) in segment {sid} that belong to the SAME chain:",
            options,
            default=default or options[0:1],
            key=f"xml_seg_{sid}",
        )
        selected[sid] = [option_to_tid[c] for c in chosen]

    return selected

def merge_tracks_xml(root, segments, selected_tracks):
    existing_ids = []
    for t in root.findall("track"):
        try:
            existing_ids.append(int(t.get("id", -1)))
        except ValueError:
            pass
    new_id = (max(existing_ids) + 1) if existing_ids else 0

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

            for box in orig.findall("box"):
                frame = int(box.get("frame", 0))
                if seg["start"] <= frame <= seg["stop"]:
                    new_track.append(deepcopy(box))

    if len(new_track.findall("box")) == 0:
        return None

    if base_label is not None:
        new_track.set("label", base_label)
    if base_group is not None:
        new_track.set("group", base_group)
    if base_source is not None:
        new_track.set("source", base_source)

    for a in reversed(base_attrs):
        new_track.insert(0, a)

    boxes = list(new_track.findall("box"))
    boxes_sorted = sorted(boxes, key=lambda b: int(b.get("frame", 0)))
    for b in boxes:
        new_track.remove(b)
    for b in boxes_sorted:
        new_track.append(b)

    for tid in used_track_ids:
        t = track_map.get(tid)
        if t is not None and t in list(root):
            root.remove(t)

    root.append(new_track)
    return root

def export_xml(root):
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return BytesIO(xml_bytes)

# -----------------------------
# Datumaro JSON helpers (2D+3D)
# -----------------------------
def is_datumaro_json(data: dict) -> bool:
    return isinstance(data, dict) and "items" in data and "categories" in data and "label" in data["categories"]

def datumaro_label_map(data: dict):
    labels = data["categories"]["label"].get("labels", [])
    return {i: lbl.get("name", f"label_{i}") for i, lbl in enumerate(labels)}

def collect_datumaro_tracks(data: dict):
    """
    returns:
      tracks[label_name][track_id] = {"count": int, "min_frame": int|None, "max_frame": int|None, "types": set[str]}
    """
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

def merge_datumaro_tracks(data: dict, label_name: str, source_track_ids, target_track_id=None):
    if not source_track_ids:
        return data
    if target_track_id is None:
        # stable default
        target_track_id = sorted(source_track_ids)[0]

    label_map = datumaro_label_map(data)
    # invert for filtering by label_name
    label_ids_for_name = {lid for lid, nm in label_map.items() if nm == label_name}

    for item in data.get("items", []):
        for ann in item.get("annotations", []):
            if ann.get("label_id") not in label_ids_for_name:
                continue
            attrs = ann.get("attributes", {}) or {}
            if attrs.get("track_id") in source_track_ids:
                attrs["track_id"] = target_track_id
                ann["attributes"] = attrs

    return data

def load_zip_find_datumaro_json(file_bytes: bytes):
    """
    Extract zip into temp dir and find the first JSON that looks like Datumaro.
    Returns (data_dict, filename_in_zip) or (None, None)
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with zipfile.ZipFile(BytesIO(file_bytes), "r") as z:
            z.extractall(td)

        json_files = list(td.rglob("*.json"))
        for jf in sorted(json_files):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            if is_datumaro_json(data):
                return data, str(jf.relative_to(td))
    return None, None

# -----------------------------
# Main
# -----------------------------
if uploaded is None:
    st.stop()

name = uploaded.name.lower()

# ---- XML path ----
if name.endswith(".xml"):
    st.header("Mode: CVAT Video XML")

    try:
        tree = ET.parse(uploaded)
        root = tree.getroot()
    except ET.ParseError as e:
        st.error(f"Failed to parse XML: {e}")
        st.stop()

    segments = parse_segments_xml(root)
    if not segments:
        segments = fallback_single_segment_from_xml(root)
        st.info("No jobs/segments found — using a single fallback segment spanning all frames.")

    job_tracks = collect_job_tracks_xml(root, segments)
    track_info = build_track_info_xml(root, segments, job_tracks)
    auto_suggestion = suggest_track_chain_xml(segments, track_info)

    st.subheader("Detected segments/jobs")
    for seg in sorted(segments, key=lambda s: s["start"]):
        sid = seg["id"]
        tracks = job_tracks.get(sid, {})
        st.write(
            f"Segment/Job {sid}: frames {seg['start']}–{seg['stop']} | tracks: {len(tracks)}"
        )

    st.markdown("---")

    selected_tracks = build_selection_ui_xml(segments, job_tracks, track_info, auto_suggestion)

    if st.button("Generate merged XML (single track)", type="primary"):
        merged_root = merge_tracks_xml(deepcopy(root), segments, selected_tracks)
        if merged_root is None:
            st.error("No valid merged track built. Check selections.")
        else:
            bio = export_xml(merged_root)
            st.success("Merged track created.")
            st.download_button(
                "Download merged CVAT XML",
                data=bio,
                file_name="merged_tracks.xml",
                mime="application/xml",
            )

# ---- JSON/ZIP path ----
else:
    st.header("Mode: Datumaro (2D/3D) Track Merge by Label + track_id")

    # Load JSON either directly or from zip
    data = None
    source_hint = None

    if name.endswith(".json"):
        try:
            data = json.load(uploaded)
            source_hint = "json"
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")
            st.stop()

    elif name.endswith(".zip"):
        try:
            raw = uploaded.getvalue()
        except Exception:
            raw = uploaded.read()

        data, found = load_zip_find_datumaro_json(raw)
        source_hint = f"zip ({found})" if found else "zip"
        if data is None:
            st.error("Could not find a Datumaro-like JSON inside the ZIP.")
            st.stop()
    else:
        st.error("Unsupported file type. Please upload .xml / .json / .zip")
        st.stop()

    if not is_datumaro_json(data):
        st.error("This JSON does not look like Datumaro format (missing 'items'/'categories/label').")
        st.stop()

    st.caption(f"Loaded Datumaro dataset from: {source_hint}")

    tracks = collect_datumaro_tracks(data)
    label_names = sorted(tracks.keys())

    if not label_names:
        st.warning("No tracked annotations found (no attributes.track_id present). Nothing to merge.")
        st.stop()

    colA, colB = st.columns([2, 3])

    with colA:
        chosen_label = st.selectbox("Choose label/class", label_names)

        # Build nice display strings "Label | track_id | frames | count | types"
        tid_to_disp = {}
        options = []
        for tid, rec in sorted(tracks[chosen_label].items(), key=lambda kv: kv[0]):
            fr = "n/a"
            if rec["min_frame"] is not None and rec["max_frame"] is not None:
                fr = f"{rec['min_frame']}-{rec['max_frame']}"
            types = ", ".join(sorted(rec["types"])) if rec["types"] else "unknown"
            disp = f"Label: {chosen_label} | track_id: {tid} | frames: {fr} | anns: {rec['count']} | types: {types}"
            tid_to_disp[disp] = tid
            options.append(disp)

        selected_disps = st.multiselect(
            "Select track_ids to merge (within this label)",
            options,
            default=options if len(options) <= 3 else options[:3],
        )

        selected_tids = [tid_to_disp[d] for d in selected_disps]

        target_mode = st.radio(
            "Target track_id",
            ["Use smallest selected", "Enter manually"],
            horizontal=True
        )
        target_tid = None
        if target_mode == "Enter manually":
            target_tid = st.number_input("Target track_id (int)", min_value=0, value=int(min(selected_tids)) if selected_tids else 0, step=1)

        merge_all_for_label = st.checkbox("Merge ALL track_ids for this label", value=False)

        run = st.button("Merge (rewrite track_id)", type="primary")

    with colB:
        st.subheader("Preview")
        st.write(f"Label: **{chosen_label}**")
        st.write(f"Tracks found: **{len(tracks[chosen_label])}**")
        st.write("Tip: Datumaro does not have CVAT jobs/segments; merging is a **track_id remap** within the chosen label.")

    if run:
        if merge_all_for_label:
            selected_tids = list(tracks[chosen_label].keys())

        if len(selected_tids) < 2:
            st.error("Select at least 2 track_ids to merge (or use 'Merge ALL').")
            st.stop()

        merged = merge_datumaro_tracks(
            deepcopy(data),
            chosen_label,
            source_track_ids=selected_tids,
            target_track_id=(None if target_mode == "Use smallest selected" else int(target_tid)),
        )

        out = json.dumps(merged, ensure_ascii=False, indent=2).encode("utf-8")
        st.success(f"Merged {len(selected_tids)} track_ids into one for label '{chosen_label}'.")
        st.download_button(
            "Download merged Datumaro JSON",
            data=out,
            file_name="merged_datumaro.json",
            mime="application/json",
        )
