import copy
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def parse_timestamp(timestamp: str) -> float:
    hh_mm_ss, ms = timestamp.split(".")
    hh, mm, ss = hh_mm_ss.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def enrich_detection_features(
    frames: List[Dict[str, Any]],
    video_metadata: Dict[str, Any],
    interval_sec: float,
) -> List[Dict[str, Any]]:
    width = int(video_metadata.get("width", 1) or 1)
    height = int(video_metadata.get("height", 1) or 1)
    fps = float(video_metadata.get("fps", 25.0) or 25.0)
    frame_area = float(max(1, width * height))

    previous_by_person: Dict[str, Dict[str, Any]] = {}

    enriched: List[Dict[str, Any]] = []
    for fr in sorted(frames, key=lambda x: int(x.get("sample_index", 0))):
        ts_sec = float(fr.get("timestamp_sec", 0.0))
        fr["interval_id"] = int(ts_sec // interval_sec)

        detections = fr.get("detections", [])
        for det in detections:
            pid = str(det.get("person_id"))
            center = det.get("center", {})
            cx = float(center.get("x", 0.0))
            cy = float(center.get("y", 0.0))

            area = float(det.get("area", 0.0))
            if area <= 0.0:
                bbox = det.get("bbox", {})
                area = float(bbox.get("w", 0.0)) * float(bbox.get("h", 0.0))

            area_ratio = area / frame_area
            area_log = float(np.log1p(max(area, 1.0)))

            motion_px = 0.0
            motion_px_per_sec = 0.0
            area_delta = 0.0
            area_ratio_delta = 0.0
            dt = 0.0

            if pid in previous_by_person:
                prev = previous_by_person[pid]
                prev_center = prev["center"]
                prev_area = float(prev["area"])
                prev_ts = float(prev["ts_sec"])
                dt = max(1.0 / fps, ts_sec - prev_ts)
                motion_px = _distance((cx, cy), prev_center)
                motion_px_per_sec = motion_px / dt
                area_delta = area - prev_area
                area_ratio_delta = area_delta / max(area, prev_area, 1.0)

            det["tracking_features"] = {
                "center_norm": {
                    "x": round(cx / width, 6),
                    "y": round(cy / height, 6),
                },
                "face_area_ratio": round(area_ratio, 6),
                "face_area_log": round(area_log, 6),
                "motion_px": round(motion_px, 4),
                "motion_px_per_sec": round(motion_px_per_sec, 4),
                "area_delta": round(area_delta, 2),
                "area_ratio_delta": round(area_ratio_delta, 6),
                "dt_sec": round(dt, 4),
            }
            det.setdefault(
                "emotion",
                {
                    "label": "unknown",
                    "confidence": None,
                    "model": "not_selected",
                },
            )

            previous_by_person[pid] = {
                "center": (cx, cy),
                "area": area,
                "ts_sec": ts_sec,
            }

        fr["face_count"] = len(detections)
        fr["person_ids"] = sorted({str(d.get("person_id")) for d in detections if d.get("person_id")})
        enriched.append(fr)

    return enriched


def link_comments(
    frames: List[Dict[str, Any]],
    comments: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    for fr in frames:
        fr.setdefault("comments", [])

    if not comments or not frames:
        return []

    frame_times = np.array([float(fr.get("timestamp_sec", 0.0)) for fr in frames], dtype=np.float32)
    linked: List[Dict[str, Any]] = []

    for idx, comment in enumerate(comments):
        if "timestamp_sec" in comment:
            ts_sec = float(comment["timestamp_sec"])
        elif "timestamp" in comment:
            ts_sec = parse_timestamp(comment["timestamp"])
        else:
            continue

        nearest_idx = int(np.argmin(np.abs(frame_times - ts_sec)))
        frame = frames[nearest_idx]
        detections = frame.get("detections", [])

        assigned_person_id = None
        if detections:
            assigned_person_id = max(
                detections,
                key=lambda d: float(d.get("area", 0.0)),
            ).get("person_id")

        record = {
            "comment_id": comment.get("comment_id", f"C{idx + 1:03d}"),
            "text": comment.get("text", ""),
            "timestamp_sec": round(ts_sec, 3),
            "timestamp": format_timestamp(ts_sec),
            "nearest_frame_index": int(frame.get("sample_index", nearest_idx)),
            "nearest_frame_timestamp": frame.get("timestamp"),
            "nearest_interval_id": int(frame.get("interval_id", -1)),
            "assigned_person_id": assigned_person_id,
        }
        frame["comments"].append(record)
        linked.append(record)

    return linked


def build_extended_intervals(
    frames: List[Dict[str, Any]],
    interval_sec: float,
) -> List[Dict[str, Any]]:
    buckets: Dict[int, Dict[str, Any]] = {}

    for fr in frames:
        ts = float(fr.get("timestamp_sec", 0.0))
        interval_id = int(fr.get("interval_id", int(ts // interval_sec)))

        if interval_id not in buckets:
            start_sec = interval_id * interval_sec
            end_sec = (interval_id + 1) * interval_sec
            buckets[interval_id] = {
                "interval_id": interval_id,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "start_ts": format_timestamp(start_sec),
                "end_ts": format_timestamp(end_sec),
                "frame_count": 0,
                "comments": [],
                "person_stats": defaultdict(
                    lambda: {
                        "frames_seen": set(),
                        "detections": 0,
                        "area_ratio_sum": 0.0,
                        "motion_sum": 0.0,
                        "motion_max": 0.0,
                        "emotion_counts": Counter(),
                    }
                ),
            }

        bucket = buckets[interval_id]
        bucket["frame_count"] += 1
        if fr.get("comments"):
            bucket["comments"].extend(fr["comments"])

        for det in fr.get("detections", []):
            pid = str(det.get("person_id"))
            tf = det.get("tracking_features", {})
            emo = det.get("emotion", {}).get("label", "unknown")

            st = bucket["person_stats"][pid]
            st["frames_seen"].add(int(fr.get("sample_index", 0)))
            st["detections"] += 1
            st["area_ratio_sum"] += float(tf.get("face_area_ratio", 0.0))
            motion = float(tf.get("motion_px_per_sec", 0.0))
            st["motion_sum"] += motion
            st["motion_max"] = max(st["motion_max"], motion)
            st["emotion_counts"][emo] += 1

    intervals: List[Dict[str, Any]] = []
    for interval_id in sorted(buckets.keys()):
        bucket = buckets[interval_id]
        frame_count = max(1, int(bucket["frame_count"]))

        person_stats_out: Dict[str, Dict[str, Any]] = {}
        for pid, st in bucket["person_stats"].items():
            detections = max(1, int(st["detections"]))
            frames_seen = len(st["frames_seen"])
            person_stats_out[pid] = {
                "frames_seen": frames_seen,
                "presence_ratio": round(frames_seen / frame_count, 4),
                "detections": int(st["detections"]),
                "avg_face_area_ratio": round(st["area_ratio_sum"] / detections, 6),
                "avg_motion_px_per_sec": round(st["motion_sum"] / detections, 4),
                "max_motion_px_per_sec": round(st["motion_max"], 4),
                "emotion_counts": dict(st["emotion_counts"]),
            }

        persons = sorted(person_stats_out.keys())
        total_faces = int(sum(v["detections"] for v in person_stats_out.values()))
        intervals.append(
            {
                "interval_id": interval_id,
                "start_sec": bucket["start_sec"],
                "end_sec": bucket["end_sec"],
                "start_ts": bucket["start_ts"],
                "end_ts": bucket["end_ts"],
                "frame_count": frame_count,
                "total_faces": total_faces,
                "avg_faces_per_frame": round(total_faces / frame_count, 4),
                "persons": persons,
                "unique_person_count": len(persons),
                "person_stats": person_stats_out,
                "comments": bucket["comments"],
            }
        )
    return intervals


def compare_intervals_extended(intervals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    comparisons: List[Dict[str, Any]] = []
    for i in range(1, len(intervals)):
        prev_it = intervals[i - 1]
        curr_it = intervals[i]

        prev_set = set(prev_it["persons"])
        curr_set = set(curr_it["persons"])
        union = prev_set | curr_set
        intersection = prev_set & curr_set
        jaccard = (len(intersection) / len(union)) if union else 1.0

        shared_deltas: List[Dict[str, Any]] = []
        for pid in sorted(intersection):
            prev_stats = prev_it["person_stats"].get(pid, {})
            curr_stats = curr_it["person_stats"].get(pid, {})
            shared_deltas.append(
                {
                    "person_id": pid,
                    "presence_ratio_delta": round(
                        float(curr_stats.get("presence_ratio", 0.0))
                        - float(prev_stats.get("presence_ratio", 0.0)),
                        4,
                    ),
                    "avg_motion_delta": round(
                        float(curr_stats.get("avg_motion_px_per_sec", 0.0))
                        - float(prev_stats.get("avg_motion_px_per_sec", 0.0)),
                        4,
                    ),
                    "avg_area_ratio_delta": round(
                        float(curr_stats.get("avg_face_area_ratio", 0.0))
                        - float(prev_stats.get("avg_face_area_ratio", 0.0)),
                        6,
                    ),
                }
            )

        comparisons.append(
            {
                "from_interval_id": prev_it["interval_id"],
                "to_interval_id": curr_it["interval_id"],
                "from_ts": prev_it["start_ts"],
                "to_ts": curr_it["start_ts"],
                "person_jaccard": round(jaccard, 4),
                "persons_joined": sorted(curr_set - prev_set),
                "persons_left": sorted(prev_set - curr_set),
                "shared_person_deltas": shared_deltas,
                "avg_faces_delta": round(
                    float(curr_it.get("avg_faces_per_frame", 0.0))
                    - float(prev_it.get("avg_faces_per_frame", 0.0)),
                    4,
                ),
                "comment_count_delta": len(curr_it.get("comments", []))
                - len(prev_it.get("comments", [])),
            }
        )
    return comparisons


def build_person_timeline(
    frames: List[Dict[str, Any]],
    person_summary: Dict[str, Dict[str, Any]],
    interval_sec: float,
) -> Dict[str, Dict[str, Any]]:
    timeline: Dict[str, Dict[str, Any]] = {}
    intervals_by_person: Dict[str, set] = defaultdict(set)
    comments_by_person: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for fr in frames:
        interval_id = int(fr.get("interval_id", 0))
        for pid in fr.get("person_ids", []):
            intervals_by_person[pid].add(interval_id)
        for c in fr.get("comments", []):
            pid = c.get("assigned_person_id")
            if pid:
                comments_by_person[pid].append(c)

    for pid, s in person_summary.items():
        frames_seen = int(s.get("frames_seen", 0))
        first_sec = float(s.get("first_seen_sec", 0.0))
        last_sec = float(s.get("last_seen_sec", first_sec))
        active_sec = max(0.0, last_sec - first_sec)
        timeline[pid] = {
            "person_id": pid,
            "first_seen_sec": round(first_sec, 3),
            "last_seen_sec": round(last_sec, 3),
            "active_window_sec": round(active_sec, 3),
            "frames_seen": frames_seen,
            "intervals_seen": sorted(intervals_by_person.get(pid, set())),
            "intervals_seen_count": len(intervals_by_person.get(pid, set())),
            "comments_count": len(comments_by_person.get(pid, [])),
            "sample_comments": comments_by_person.get(pid, [])[:5],
        }
    return dict(sorted(timeline.items()))


def emotion_model_review() -> Dict[str, Any]:
    candidates = [
        {
            "name": "fer (MiniXception, FER2013)",
            "package": "fer",
            "accuracy": "medium",
            "speed": "high",
            "cpu_friendly": True,
            "notes": "Fast baseline for quick iteration on notebooks.",
        },
        {
            "name": "DeepFace (emotion backend)",
            "package": "deepface",
            "accuracy": "medium_high",
            "speed": "medium_low",
            "cpu_friendly": False,
            "notes": "Good quality, heavier dependencies.",
        },
        {
            "name": "HSEmotion (ONNX/PyTorch variants)",
            "package": "hsemotion",
            "accuracy": "high",
            "speed": "medium",
            "cpu_friendly": True,
            "notes": "Strong robustness to pose changes, good next target.",
        },
    ]

    recommendation = {
        "phase_1": {
            "model": "fer (MiniXception, FER2013)",
            "why": "Fast to integrate and benchmark on current pipeline.",
        },
        "phase_2": {
            "model": "HSEmotion",
            "why": "Better stability for moving client and profile faces.",
        },
        "benchmark_plan": [
            "Run on same consolidated face crops for 2 speakers.",
            "Measure per-person label stability over intervals.",
            "Track FPS/latency per 100 crops on CPU.",
            "Choose model by stability + speed trade-off.",
        ],
    }
    return {"candidates": candidates, "recommendation": recommendation}


def run_step3(
    input_path: str = "extracted_frames_v2/analysis_step2_consolidated.json",
    output_path: str = "extracted_frames_v2/analysis_step3_extended.json",
    interval_sec: float = 10.0,
    comments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    data = load_json(input_path)
    result = copy.deepcopy(data)

    frames = result.get("frames", [])
    video_metadata = result.get("video_metadata", {})

    frames = enrich_detection_features(
        frames=frames,
        video_metadata=video_metadata,
        interval_sec=interval_sec,
    )
    linked_comments = link_comments(frames, comments=comments)

    intervals_ext = build_extended_intervals(frames, interval_sec=interval_sec)
    interval_cmp_ext = compare_intervals_extended(intervals_ext)

    person_summary = result.get("person_summary", {})
    person_timeline = build_person_timeline(
        frames=frames,
        person_summary=person_summary,
        interval_sec=interval_sec,
    )

    result["step3_config"] = {
        "interval_sec": interval_sec,
        "features": [
            "tracking_features",
            "interval_person_stats",
            "interval_comparison_extended",
            "comment_to_person_linking",
        ],
    }
    result["frames"] = frames
    result["intervals_extended"] = intervals_ext
    result["interval_comparison_extended"] = interval_cmp_ext
    result["linked_comments"] = linked_comments
    result["person_timeline"] = person_timeline
    result["emotion_model_review"] = emotion_model_review()

    save_json(result, output_path)

    print("=== STEP 3: EXTENDED ANALYTICS + EMOTION MODEL REVIEW ===")
    print(f"Frames: {len(frames)}")
    print(f"Intervals (extended): {len(intervals_ext)}")
    print(f"Linked comments: {len(linked_comments)}")
    print(f"Persons in timeline: {len(person_timeline)} -> {list(person_timeline.keys())}")
    print(f"Saved: {output_path}")

    return result


if __name__ == "__main__":
    run_step3()
