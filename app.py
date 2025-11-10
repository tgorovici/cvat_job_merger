import streamlit as st
import xml.etree.ElementTree as ET
from io import BytesIO
from copy import deepcopy
import math

st.set_page_config(page_title="CVAT Track Merger", layout="wide")

st.title("CVAT Video XML Track Merger")

st.markdown(
    """
Upload a **CVAT Video XML** (with segments/jobs). This app:

1. Detects jobs/segments from `<meta>`.
2. Analyzes tracks and their box coordinates.
3. **Suggests** one continuous track across segments:
   - Single-track segments → auto-selected.
   - Multi-track segments → picks the closest continuation by coordinates (with label/attribute checks).
4. Lets you adjust selections per segment.
5. Exports a new CVAT Video XML with one merged track (removing the originals used for the merge).
"""
)

uploaded_file = st.file_uploader("Upload CVAT video XML", type=["xml"])

# ===== Helpers =====

def parse_segments(root):
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


def collect_job_tracks(root, segments):
    """
    Map which tracks appear in each segment's frame range.
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


def get_track_boxes_by_frame(track):
    """
    Return sorted list of (frame, cx, cy) for all boxes.
    """
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


def build_track_info(root, segments, job_tracks):
    """
    Enrich job_tracks with geometry and frame stats for continuity scoring.
    """
    track_map = {t.get("id"): t for t in root.findall("track")}
    info = {seg["id"]: {} for seg in segments}

    for seg in segments:
        sid = seg["id"]
        for tid, meta in job_tracks.get(sid, {}).items():
            t = track_map.get(tid)
            if t is None:
                continue
            boxes = get_track_boxes_by_frame(t)
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


def attr_signature(attrs):
    return tuple(sorted(attrs.items()))


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def suggest_track_chain(segments, track_info):
    """
    Suggest exactly ONE track per segment that forms a continuous object.

    Strategy:
    - If only one track in segment -> select it.
    - If multiple:
        * If there is a previous chosen track:
            - prefer same label
            - prefer same attributes
            - minimize distance between:
                prev segment last_box center -> candidate first_box center
        * Otherwise fallback = longest track in that segment.
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

        # Single option
        if len(candidates) == 1:
            tid = next(iter(candidates.keys()))
            suggestion[sid] = [tid]
            prev_tid, prev_sid = tid, sid
            continue

        # First ambiguous segment with no history: choose longest
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

        # Score each candidate
        for tid, ci in candidates.items():
            label = ci["label"]
            cand_center = (ci["first_box"][1], ci["first_box"][2])
            score = 0.0

            # Require same label if possible
            if label != prev_label:
                score += 10_000  # big penalty

            # Attribute mismatch penalty
            if attr_signature(ci["attributes"]) != prev_attr_sig:
                score += 1_000

            # Coordinate distance
            score += euclidean(prev_center, cand_center)

            if score < best_score:
                best_score = score
                best_tid = tid

        # Fallback: if everything penalized, still pick min-score or longest
        if best_tid is None:
            best_tid = max(
                candidates.items(),
                key=lambda kv: kv[1]["end_frame"] - kv[1]["start_frame"]
            )[0]

        suggestion[sid] = [best_tid]
        prev_tid, prev_sid = best_tid, sid

    return suggestion


def build_selection_ui_with_suggestions(segments, job_tracks, track_info, auto_suggestion):
    st.subheader("Track selection per segment (you can override suggestions)")
    selected = {}

    for seg in sorted(segments, key=lambda s: s["start"]):
        sid = seg["id"]
        tracks = job_tracks.get(sid, {})
        if not tracks:
            continue

        st.markdown(f"**Segment / Job {sid}**")

        options = []
        option_to_tid = {}
        default = []

        for tid, meta in tracks.items():
            ti = track_info.get(sid, {}).get(tid, {})
            attr_str = ", ".join(f"{k}={v}" for k, v in meta["attributes"].items()) or "no attributes"
            sf, ef = ti.get("start_frame"), ti.get("end_frame")
            fr_str = f"{sf}-{ef}" if sf is not None and ef is not None else "n/a"
            desc = f"Track {tid} | Label: {meta['label']} | {attr_str} | Frames: {fr_str}"

            if auto_suggestion.get(sid) == [tid]:
                desc += "  ⟵ suggested"
                default = [desc]

            options.append(desc)
            option_to_tid[desc] = tid

        chosen = st.multiselect(
            f"Choose track(s) in segment {sid} to belong to the SAME object chain:",
            options,
            default=default or options[0:1],
            key=f"seg_{sid}",
        )

        selected[sid] = [option_to_tid[c] for c in chosen]

    return selected


def merge_tracks(root, segments, selected_tracks):
    """
    Build one merged <track> from selected tracks across segments.
    """
    # Find free ID
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

    # Collect boxes in segment order
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

            # Add boxes that fall inside this segment
            for box in orig.findall("box"):
                frame = int(box.get("frame", 0))
                if seg["start"] <= frame <= seg["stop"]:
                    new_track.append(deepcopy(box))

    if len(new_track.findall("box")) == 0:
        return None

    # Set merged properties
    if base_label is not None:
        new_track.set("label", base_label)
    if base_group is not None:
        new_track.set("group", base_group)
    if base_source is not None:
        new_track.set("source", base_source)

    for a in base_attrs:
        new_track.insert(0, a)

    # Sort boxes by frame
    boxes = list(new_track.findall("box"))
    boxes_sorted = sorted(boxes, key=lambda b: int(b.get("frame", 0)))
    for b in boxes:
        new_track.remove(b)
    for b in boxes_sorted:
        new_track.append(b)

    # Drop original tracks that contributed
    for tid in used_track_ids:
        t = track_map.get(tid)
        if t is not None and t in root:
            root.remove(t)

    root.append(new_track)
    return root


def export_xml(root):
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return BytesIO(xml_bytes)

# ===== Main flow =====

if uploaded_file is not None:
    try:
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
    except ET.ParseError as e:
        st.error(f"Failed to parse XML: {e}")
    else:
        segments = parse_segments(root)
        if not segments:
            st.warning("No segments/jobs found in <meta>. Make sure this is a CVAT **video** XML with segments or job frame ranges.")
        else:
            job_tracks = collect_job_tracks(root, segments)
            track_info = build_track_info(root, segments, job_tracks)
            auto_suggestion = suggest_track_chain(segments, track_info)

            st.subheader("Detected segments and tracks")

            for seg in sorted(segments, key=lambda s: s["start"]):
                sid = seg["id"]
                tracks = job_tracks.get(sid, {})
                if tracks:
                    st.write(
                        f"Segment {sid}: frames {seg['start']}–{seg['stop']} | "
                        f"{len(tracks)} track(s): {', '.join(tracks.keys())}"
                    )
                else:
                    st.write(
                        f"Segment {sid}: frames {seg['start']}–{seg['stop']} | no tracks"
                    )

            st.markdown("---")

            # Build selection UI with coordinate-based suggested chain
            selected_tracks = build_selection_ui_with_suggestions(
                segments, job_tracks, track_info, auto_suggestion
            )

            if st.button("Generate merged XML", type="primary"):
                merged_root = merge_tracks(deepcopy(root), segments, selected_tracks)
                if merged_root is None:
                    st.error("No valid merged track built. Check selections.")
                else:
                    bio = export_xml(merged_root)
                    st.success("Merged track created. Download below.")
                    st.download_button(
                        "Download merged CVAT video XML",
                        data=bio,
                        file_name="merged_tracks.xml",
                        mime="application/xml",
                    )
