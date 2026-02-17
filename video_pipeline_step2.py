import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackState:
    person_id: str
    last_embedding: np.ndarray
    last_center: Tuple[float, float]
    last_area: float
    last_sample_index: int
    hits: int = 0


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


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_image_path(manifest_path: str, image_path: str) -> str:
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    if os.path.exists(image_path):
        return image_path

    # Frames from step1 are in the same folder as manifest.
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    candidate = os.path.join(manifest_dir, os.path.basename(image_path))
    if os.path.exists(candidate):
        return candidate
    return image_path


def build_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return detector


def detect_faces(
    img_bgr: np.ndarray,
    detector: cv2.CascadeClassifier,
    scale_factor: float = 1.05,
    min_neighbors: int = 3,
    min_size: Tuple[int, int] = (30, 30),
) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def face_embedding(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x, y, bw, bh = clamp_bbox(bbox, w, h)
    roi = img_bgr[y : y + bh, x : x + bw]
    if roi.size == 0:
        return np.zeros((288,), dtype=np.float32)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gray_vec = gray_small.flatten()  # 256 dims

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_h = hist_h / (np.sum(hist_h) + 1e-6)
    hist_s = hist_s / (np.sum(hist_s) + 1e-6)

    emb = np.concatenate([gray_vec, hist_h, hist_s]).astype(np.float32)
    norm = np.linalg.norm(emb) + 1e-6
    return emb / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def center_of(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def area_of(bbox: Tuple[int, int, int, int]) -> float:
    _, _, w, h = bbox
    return float(w * h)


def match_score(
    det_embedding: np.ndarray,
    det_center: Tuple[float, float],
    det_area: float,
    track: TrackState,
    frame_diag: float,
) -> float:
    app_dist = cosine_distance(det_embedding, track.last_embedding)
    spatial_dist = (
        np.linalg.norm(np.array(det_center) - np.array(track.last_center)) / max(frame_diag, 1.0)
    )
    area_dist = abs(det_area - track.last_area) / max(det_area, track.last_area, 1.0)
    return 0.70 * app_dist + 0.20 * spatial_dist + 0.10 * area_dist


def assign_person_ids(
    detections: List[Tuple[int, int, int, int]],
    embeddings: List[np.ndarray],
    sample_index: int,
    frame_shape: Tuple[int, int, int],
    tracks: Dict[str, TrackState],
    next_person_num: int,
    max_inactive_frames: int = 8,
    threshold: float = 0.45,
) -> Tuple[List[Dict[str, Any]], int]:
    h, w = frame_shape[:2]
    frame_diag = float((h ** 2 + w ** 2) ** 0.5)

    active_tracks: Dict[str, TrackState] = {}
    for pid, tr in tracks.items():
        if sample_index - tr.last_sample_index <= max_inactive_frames:
            active_tracks[pid] = tr

    used_tracks = set()
    results: List[Dict[str, Any]] = []

    for det_idx, bbox in enumerate(detections):
        emb = embeddings[det_idx]
        det_center = center_of(bbox)
        det_area = area_of(bbox)

        best_pid = None
        best_score = 10.0

        for pid, tr in active_tracks.items():
            if pid in used_tracks:
                continue
            score = match_score(emb, det_center, det_area, tr, frame_diag)
            if score < best_score:
                best_score = score
                best_pid = pid

        if best_pid is None or best_score > threshold:
            next_person_num += 1
            best_pid = f"P{next_person_num:03d}"
            best_score = 0.0

        used_tracks.add(best_pid)

        tracks[best_pid] = TrackState(
            person_id=best_pid,
            last_embedding=emb,
            last_center=det_center,
            last_area=det_area,
            last_sample_index=sample_index,
            hits=tracks[best_pid].hits + 1 if best_pid in tracks else 1,
        )

        x, y, bw, bh = bbox
        results.append(
            {
                "face_idx": det_idx,
                "person_id": best_pid,
                "bbox": {"x": x, "y": y, "w": bw, "h": bh},
                "center": {"x": round(det_center[0], 2), "y": round(det_center[1], 2)},
                "area": int(det_area),
                "match_score": round(float(best_score), 4),
                "confidence": 1.0,
                "embedding_preview": [round(float(v), 5) for v in emb[:8]],
            }
        )

    return results, next_person_num


def attach_comments_to_frames(
    frame_records: List[Dict[str, Any]],
    comments: Optional[List[Dict[str, Any]]],
) -> None:
    if not comments:
        return
    if not frame_records:
        return

    frame_times = np.array([float(fr.get("timestamp_sec", 0.0)) for fr in frame_records], dtype=np.float32)

    for idx, comment in enumerate(comments):
        if "timestamp_sec" in comment:
            ts_sec = float(comment["timestamp_sec"])
        elif "timestamp" in comment:
            ts_sec = parse_timestamp(comment["timestamp"])
        else:
            continue

        nearest_idx = int(np.argmin(np.abs(frame_times - ts_sec)))
        target = frame_records[nearest_idx]
        target.setdefault("comments", [])
        target["comments"].append(
            {
                "comment_id": comment.get("comment_id", f"C{idx+1:03d}"),
                "text": comment.get("text", ""),
                "timestamp_sec": round(ts_sec, 3),
                "timestamp": format_timestamp(ts_sec),
            }
        )


def aggregate_intervals(
    frame_records: List[Dict[str, Any]],
    interval_sec: float,
) -> List[Dict[str, Any]]:
    buckets: Dict[int, Dict[str, Any]] = {}

    for fr in frame_records:
        ts = float(fr.get("timestamp_sec", 0.0))
        interval_id = int(ts // interval_sec)
        fr["interval_id"] = interval_id

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
                "total_faces": 0,
                "persons": set(),
                "person_frame_hits": defaultdict(int),
                "comments": [],
            }

        bucket = buckets[interval_id]
        bucket["frame_count"] += 1
        face_count = int(fr.get("face_count", 0))
        bucket["total_faces"] += face_count
        for pid in fr.get("person_ids", []):
            bucket["persons"].add(pid)
            bucket["person_frame_hits"][pid] += 1
        if fr.get("comments"):
            bucket["comments"].extend(fr["comments"])

    intervals: List[Dict[str, Any]] = []
    for interval_id in sorted(buckets.keys()):
        bucket = buckets[interval_id]
        persons_sorted = sorted(bucket["persons"])
        frame_count = max(1, bucket["frame_count"])
        presence_ratio = {
            pid: round(bucket["person_frame_hits"][pid] / frame_count, 3)
            for pid in persons_sorted
        }

        intervals.append(
            {
                "interval_id": interval_id,
                "start_sec": bucket["start_sec"],
                "end_sec": bucket["end_sec"],
                "start_ts": bucket["start_ts"],
                "end_ts": bucket["end_ts"],
                "frame_count": bucket["frame_count"],
                "total_faces": bucket["total_faces"],
                "avg_faces_per_frame": round(bucket["total_faces"] / frame_count, 3),
                "persons": persons_sorted,
                "unique_person_count": len(persons_sorted),
                "person_presence_ratio": presence_ratio,
                "comments": bucket["comments"],
            }
        )

    return intervals


def compare_intervals(intervals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    comparisons: List[Dict[str, Any]] = []
    for i in range(1, len(intervals)):
        prev_it = intervals[i - 1]
        curr_it = intervals[i]
        prev_set = set(prev_it["persons"])
        curr_set = set(curr_it["persons"])

        union = prev_set | curr_set
        intersection = prev_set & curr_set
        jaccard = (len(intersection) / len(union)) if union else 1.0

        comparisons.append(
            {
                "from_interval_id": prev_it["interval_id"],
                "to_interval_id": curr_it["interval_id"],
                "from_ts": prev_it["start_ts"],
                "to_ts": curr_it["start_ts"],
                "persons_joined": sorted(curr_set - prev_set),
                "persons_left": sorted(prev_set - curr_set),
                "shared_persons": sorted(intersection),
                "person_jaccard": round(jaccard, 3),
                "avg_faces_delta": round(
                    curr_it["avg_faces_per_frame"] - prev_it["avg_faces_per_frame"], 3
                ),
                "face_count_delta": curr_it["total_faces"] - prev_it["total_faces"],
            }
        )
    return comparisons


def summarize_persons(frame_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}

    for fr in frame_records:
        ts = float(fr["timestamp_sec"])
        ts_txt = fr["timestamp"]
        for det in fr.get("detections", []):
            pid = det["person_id"]
            if pid not in summary:
                summary[pid] = {
                    "person_id": pid,
                    "first_seen_sec": ts,
                    "last_seen_sec": ts,
                    "first_seen_ts": ts_txt,
                    "last_seen_ts": ts_txt,
                    "detections_count": 0,
                    "frames_seen": 0,
                    "area_sum": 0.0,
                    "sample_crops": [],
                }
            p = summary[pid]
            p["detections_count"] += 1
            p["area_sum"] += float(det["area"])
            p["last_seen_sec"] = ts
            p["last_seen_ts"] = ts_txt
            if p["first_seen_sec"] > ts:
                p["first_seen_sec"] = ts
                p["first_seen_ts"] = ts_txt
            if det.get("crop_path") and len(p["sample_crops"]) < 5:
                p["sample_crops"].append(det["crop_path"])

    frames_per_person = defaultdict(set)
    for fr in frame_records:
        for pid in fr.get("person_ids", []):
            frames_per_person[pid].add(fr["sample_index"])

    for pid, person in summary.items():
        person["frames_seen"] = len(frames_per_person[pid])
        person["avg_area"] = round(person["area_sum"] / max(1, person["detections_count"]), 2)
        person["first_seen_sec"] = round(person["first_seen_sec"], 3)
        person["last_seen_sec"] = round(person["last_seen_sec"], 3)
        del person["area_sum"]

    return dict(sorted(summary.items()))


def run_step2(
    manifest_path: str = "extracted_frames_v2/frame_manifest_step1.json",
    output_path: str = "extracted_frames_v2/analysis_step2.json",
    interval_sec: float = 10.0,
    comments: Optional[List[Dict[str, Any]]] = None,
    save_face_crops: bool = True,
    crops_dir: str = "extracted_frames_v2/faces_by_person",
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    frame_records: List[Dict[str, Any]] = manifest.get("frames", [])
    attach_comments_to_frames(frame_records, comments)

    detector = build_detector()
    if detector.empty():
        raise RuntimeError("Haar Cascade не загружен.")

    if save_face_crops:
        os.makedirs(crops_dir, exist_ok=True)

    tracks: Dict[str, TrackState] = {}
    next_person_num = 0

    enriched_frames: List[Dict[str, Any]] = []
    for fr in frame_records:
        sample_index = int(fr["sample_index"])
        img_path = resolve_image_path(manifest_path, fr["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            fr["detections"] = []
            fr["face_count"] = 0
            fr["person_ids"] = []
            enriched_frames.append(fr)
            continue

        detections = detect_faces(img, detector=detector)
        embeddings = [face_embedding(img, bbox) for bbox in detections]

        assigned, next_person_num = assign_person_ids(
            detections=detections,
            embeddings=embeddings,
            sample_index=sample_index,
            frame_shape=img.shape,
            tracks=tracks,
            next_person_num=next_person_num,
        )

        if save_face_crops:
            ih, iw = img.shape[:2]
            for det in assigned:
                pid = det["person_id"]
                x = det["bbox"]["x"]
                y = det["bbox"]["y"]
                bw = det["bbox"]["w"]
                bh = det["bbox"]["h"]
                x, y, bw, bh = clamp_bbox((x, y, bw, bh), iw, ih)
                crop = img[y : y + bh, x : x + bw]
                person_dir = os.path.join(crops_dir, pid)
                os.makedirs(person_dir, exist_ok=True)
                crop_name = f"s{sample_index:05d}_f{det['face_idx']:02d}.jpg"
                crop_path = os.path.join(person_dir, crop_name)
                cv2.imwrite(crop_path, crop)
                det["crop_path"] = crop_path

        fr["detections"] = assigned
        fr["face_count"] = len(assigned)
        fr["person_ids"] = sorted({d["person_id"] for d in assigned})
        enriched_frames.append(fr)

    intervals = aggregate_intervals(enriched_frames, interval_sec=interval_sec)
    interval_comparison = compare_intervals(intervals)
    person_summary = summarize_persons(enriched_frames)

    result = {
        "video_metadata": manifest.get("video_metadata", {}),
        "step2_config": {
            "interval_sec": interval_sec,
            "detector": "haar_frontalface_default",
            "tracking": "appearance+spatial matching",
            "comments_linked": bool(comments),
        },
        "frames": enriched_frames,
        "intervals": intervals,
        "interval_comparison": interval_comparison,
        "person_summary": person_summary,
    }
    save_json(result, output_path)

    total_faces = int(sum(fr.get("face_count", 0) for fr in enriched_frames))
    print("=== STEP 2: TRACKING + INTERVAL ANALYSIS ===")
    print(f"Frames processed: {len(enriched_frames)}")
    print(f"Total face detections: {total_faces}")
    print(f"Unique person IDs: {len(person_summary)} -> {list(person_summary.keys())}")
    print(f"Intervals: {len(intervals)} (interval_sec={interval_sec})")
    print(f"Saved analysis: {output_path}")

    return result


if __name__ == "__main__":
    run_step2()
