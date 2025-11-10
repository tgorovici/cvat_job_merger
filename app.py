import streamlit as st
import xml.etree.ElementTree as ET
from io import BytesIO
from copy import deepcopy

st.set_page_config(page_title="CVAT Track Merger", layout="wide")

st.title("CVAT Video XML Track Merger")

st.markdown(
    """
Upload a **CVAT Video XML** (with jobs/segments) and:

1. If each job contains a *single track*, automatically merge them into one continuous track.
2. If some jobs contain **multiple tracks**, select which tracks to merge into one.

Then download the updated CVAT video XML with a single merged track.
"""
)

uploaded_file = st.file_uploader("Upload CVAT video XML", type=["xml"])

# Session state
if "parsed" not in st.session_state:
    st.session_state.parsed = None
if "segments" not in st.session_state:
    st.session_state.segments = []
if "job_tracks" not in st.session_state:
    st.session_state.job_tracks = {}
if "selected_tracks" not in st.session_state:
    st.session_state.selected_tracks = {}
if "auto_merge_possible" not in st.session_state:
    st.session_state.auto_merge_possible = False


def parse_segments(root):
    """
    Extract job/segment definitions from the CVAT meta.

    Supports:
    - <meta><task><segments><segment>...</segment></segments></task>
    - <meta><job>...</job> (single-job export)

    Returns:
        list of {id, start, stop, source}
    """
    segments = []

    meta = root.find("meta")
    if meta is None:
        return segments

    # Case 1: Task with segments (common multi-job video export)
    task = meta.find("task")
    if task is not None:
        seg_container = task.find("segments")
        if seg_container is not None:
            for seg in seg_container.findall("segment"):
                seg_id = seg.findtext("id")
                start = seg.findtext("start")
                stop = seg.findtext("stop")
                if start is not None and stop is not None:
                    segments.append(
                        {
                            "id": seg_id if seg_id is not None else str(len(segments)),
                            "start": int(start),
                            "stop": int(stop),
                            "source": "segment",
                        }
                    )

    # Case 2: Simple job export
    job = meta.find("job")
    if job is not None:
        start = job.findtext("start_frame")
        stop = job.findtext("stop_frame")
        if start is not None and stop is not None:
            segments.append(
                {
                    "id": job.findtext("id") or "0",
                    "start": int(start),
                    "stop": int(stop),
                    "source": "job",
                }
            )

    return segments


def collect_job_tracks(root, segments):
    """
    For each segment, find which tracks have boxes in its frame range.

    Returns:
        job_tracks: {
            seg_id: {
                track_id: {
                    'label': str,
                    'attributes': {name: value},
                    'frames': [frame_numbers...]
                }, ...
            }, ...
        }
    """
    job_tracks = {seg["id"]: {} for seg in segments}

    for track in root.findall("track"):
        track_id = track.get("id")
        label = track.get("label", "")

        # Track-level attributes
        attrs = {}
        for attr in track.findall("attribute"):
            name = attr.get("name")
            if name is not None:
                attrs[name] = (attr.text or "").strip()

        # Check all boxes of this track
        for box in track.findall("box"):
            frame = int(box.get("frame", 0))
            for seg in segments:
                if seg["start"] <= frame <= seg["stop"]:
                    seg_tracks = job_tracks.setdefault(seg["id"], {})
                    entry = seg_tracks.setdefault(
                        track_id,
                        {"label": label, "attributes": attrs, "frames": []},
                    )
                    entry["frames"].append(frame)

    return job_tracks


def can_auto_merge(job_tracks):
    """
    Auto-merge is allowed only if:
    - Every segment that has any tracks has EXACTLY ONE candidate track.
    """
    has_any = False
    for _, tracks in job_tracks.items():
        if tracks:
            has_any = True
            if len(tracks) != 1:
                return False
    return has_any


def build_selection_ui(job_tracks):
    """
    Build the manual selection UI.

    Returns:
        selected: {seg_id: [track_ids_to_merge]}
    """
    st.subheader("Select tracks to merge (Manual Mode)")
    st.markdown(
        "For each job/segment, choose which track(s) belong to the same physical object and should be merged."
    )

    selected = {}
    for seg_id, tracks in job_tracks.items():
        if not tracks:
            continue

        st.markdown(f"**Job / Segment {seg_id}**")

        options = []
        option_to_tid = {}

        for tid, info in tracks.items():
            attr_str = ", ".join(
                f"{k}={v}" for k, v in info["attributes"].items()
            ) or "no attributes"
            frame_range = f"{min(info['frames'])}-{max(info['frames'])}" if info["frames"] else "n/a"
            desc = f"Track {tid} | Label: {info['label']} | {attr_str} | Frames: {frame_range}"
            options.append(desc)
            option_to_tid[desc] = tid

        chosen = st.multiselect(
            f"Tracks in segment {seg_id}",
            options,
            default=options[0:1],
            key=f"seg_{seg_id}",
        )

        selected[seg_id] = [option_to_tid[c] for c in chosen]

    return selected


def merge_tracks(root, segments, selected_tracks):
    """
    Merge the chosen tracks into a single new track.

    Rules:
    - Process segments in ascending start frame.
    - For each segment, take boxes from the selected track IDs
      that fall within that segment's frame span.
    - Use label/group/source + attributes from the first contributing track.
    - Remove all original tracks that contributed.
    - Append one new merged <track> at the end.
    """
    # Allocate new track id
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

    all_selected_track_ids = set()

    # Collect boxes in chronological order of segments
    for seg in sorted(segments, key=lambda s: s["start"]):
        seg_id = seg["id"]
        ids_to_merge = selected_tracks.get(seg_id, [])
        if not ids_to_merge:
            continue

        for tid in ids_to_merge:
            orig = track_map.get(tid)
            if orig is None:
                continue

            all_selected_track_ids.add(tid)

            # Initialize merged track properties from first valid track
            if base_label is None:
                base_label = orig.get("label", "")
                base_group = orig.get("group", "0")
                base_source = orig.get("source", "manual")
                for a in orig.findall("attribute"):
                    base_attrs.append(deepcopy(a))

            # Add boxes in this segment range
            for box in orig.findall("box"):
                frame = int(box.get("frame", 0))
                if seg["start"] <= frame <= seg["stop"]:
                    new_track.append(deepcopy(box))

    # If no boxes collected -> nothing to merge
    if len(new_track.findall("box")) == 0:
        return None

    # Set merged track properties
    if base_label is not None:
        new_track.set("label", base_label)
    if base_group is not None:
        new_track.set("group", base_group)
    if base_source is not None:
        new_track.set("source", base_source)

    # Add attributes at top of track
    for a in base_attrs:
        new_track.insert(0, a)

    # Sort boxes by frame index
    boxes = new_track.findall("box")
    boxes_sorted = sorted(boxes, key=lambda b: int(b.get("frame", 0)))
    for b in boxes:
        new_track.remove(b)
    for b in boxes_sorted:
        new_track.append(b)

    # Remove all original contributing tracks
    for tid in all_selected_track_ids:
        t = track_map.get(tid)
        if t is not None and t in root:
            root.remove(t)

    # Append the merged track
    root.append(new_track)

    return root


def export_xml(root):
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return BytesIO(xml_bytes)


# === Main logic ===
if uploaded_file is not None:
    try:
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
    except ET.ParseError as e:
        st.error(f"Failed to parse XML: {e}")
    else:
        segments = parse_segments(root)

        if not segments:
            st.warning(
                "No jobs/segments found under <meta>. "
                "This tool expects CVAT **video** XML with segments or job frame ranges."
            )
        else:
            job_tracks = collect_job_tracks(root, segments)

            st.session_state.parsed = root
            st.session_state.segments = segments
            st.session_state.job_tracks = job_tracks

            # Display summary
            st.subheader("Detected segments and tracks")
            for seg in segments:
                seg_id = seg["id"]
                tracks = job_tracks.get(seg_id, {})
                if tracks:
                    st.write(
                        f"Segment {seg_id}: frames {seg['start']}–{seg['stop']} | "
                        f"{len(tracks)} track(s): {', '.join(tracks.keys())}"
                    )
                else:
                    st.write(
                        f"Segment {seg_id}: frames {seg['start']}–{seg['stop']} | no tracks"
                    )

            auto_possible = can_auto_merge(job_tracks)
            st.session_state.auto_merge_possible = auto_possible

            merge_mode = st.radio(
                "Merge mode",
                (
                    "Auto: merge single-track jobs into one track",
                    "Manual: choose tracks per job to merge",
                ),
            )

            # AUTO MODE
            if merge_mode.startswith("Auto"):
                if not auto_possible:
                    st.warning(
                        "Auto merge is not possible: at least one segment has zero or multiple tracks.\n"
                        "Switch to manual mode and choose the correct track per segment."
                    )
                else:
                    # Build mapping: for each non-empty segment, use its single track
                    selected_tracks = {}
                    for seg_id, tracks in job_tracks.items():
                        if len(tracks) == 1:
                            (tid,) = tracks.keys()
                            selected_tracks[seg_id] = [tid]

                    st.session_state.selected_tracks = selected_tracks

                    if st.button("Generate merged XML", type="primary"):
                        merged_root = merge_tracks(
                            deepcopy(st.session_state.parsed),
                            st.session_state.segments,
                            st.session_state.selected_tracks,
                        )
                        if merged_root is None:
                            st.error("No boxes found to merge. Check your XML content.")
                        else:
                            bio = export_xml(merged_root)
                            st.success("Merged track created successfully.")
                            st.download_button(
                                "Download merged CVAT video XML",
                                data=bio,
                                file_name="merged_tracks.xml",
                                mime="application/xml",
                            )

            # MANUAL MODE
            else:
                selected_tracks = build_selection_ui(job_tracks)
                st.session_state.selected_tracks = selected_tracks

                if st.button("Generate merged XML from selected tracks", type="primary"):
                    merged_root = merge_tracks(
                        deepcopy(st.session_state.parsed),
                        st.session_state.segments,
                        st.session_state.selected_tracks,
                    )
                    if merged_root is None:
                        st.error(
                            "No valid selections or no boxes found in chosen tracks. "
                            "Make sure you picked the correct track(s) per segment."
                        )
                    else:
                        bio = export_xml(merged_root)
                        st.success("Merged track created successfully from selected tracks.")
                        st.download_button(
                            "Download merged CVAT video XML",
                            data=bio,
                            file_name="merged_tracks.xml",
                            mime="application/xml",
                        )
