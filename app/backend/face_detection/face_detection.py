"""
Face detection module: frame extraction, detection, tracking, enrichment.
Merged from pipeline steps 1–4.
"""
import copy
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# --- Shared types and helpers (deduplicated from steps 1–4) ---

@dataclass
class VideoMetadata:
    video_path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: Optional[float]


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


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Step 1: Video metadata and frame extraction ---

def read_video_metadata(video_path: str) -> VideoMetadata:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(raw_fps) if raw_fps and raw_fps > 0 else 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = (frame_count / fps) if frame_count > 0 and fps > 0 else None
    cap.release()
    return VideoMetadata(
        video_path=video_path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def build_sampling_plan(
    metadata: VideoMetadata,
    interval_sec: float = 0.5,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> List[int]:
    if metadata.frame_count <= 0:
        return []
    fps = metadata.fps
    step_frames = max(1, int(round(interval_sec * fps)))
    start_frame = max(0, int(round(start_sec * fps)))
    if end_sec is None:
        end_frame = metadata.frame_count - 1
    else:
        end_frame = min(metadata.frame_count - 1, int(round(end_sec * fps)))
    frame_indices = list(range(start_frame, end_frame + 1, step_frames))
    if max_frames is not None and max_frames > 0:
        frame_indices = frame_indices[:max_frames]
    return frame_indices


def extract_frames_with_timestamps(
    video_path: str,
    frame_indices: List[int],
    metadata: VideoMetadata,
    output_dir: str = "extracted_frames_v2",
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    records: List[Dict[str, Any]] = []
    for sample_idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        timestamp_sec = frame_idx / metadata.fps
        file_name = f"frame_{sample_idx:05d}_f{frame_idx:06d}.jpg"
        file_path = os.path.join(output_dir, file_name)
        cv2.imwrite(file_path, frame)
        records.append({
            "sample_index": sample_idx,
            "frame_index": frame_idx,
            "timestamp_sec": round(timestamp_sec, 6),
            "timestamp": format_timestamp(timestamp_sec),
            "image_path": file_path,
            "interval_id": None,
            "person_ids": [],
            "comments": [],
        })
    cap.release()
    return records


def save_step1_manifest(
    metadata: VideoMetadata,
    frame_records: List[Dict[str, Any]],
    output_dir: str = "extracted_frames_v2",
    manifest_path: Optional[str] = None,
) -> str:
    manifest = {
        "video_metadata": asdict(metadata),
        "sampling": {"total_samples": len(frame_records), "fps_based": True},
        "frames": frame_records,
    }
    path = manifest_path or os.path.join(output_dir, "frame_manifest_step1.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def run_step1(
    video_path: str = "video2.webm",
    interval_sec: float = 0.5,
    max_frames: Optional[int] = 200,
    output_dir: str = "extracted_frames_v2",
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    metadata = read_video_metadata(video_path)
    print("=== STEP 1: VIDEO METADATA ===")
    print(f"Video: {metadata.video_path}")
    print(f"FPS (from metadata): {metadata.fps:.3f}")
    print(f"Total frames: {metadata.frame_count}")
    print(f"Resolution: {metadata.width}x{metadata.height}")
    if metadata.duration_sec is not None:
        print(f"Duration: {metadata.duration_sec:.2f} sec")
    frame_indices = build_sampling_plan(
        metadata=metadata, interval_sec=interval_sec, start_sec=0.0, end_sec=None, max_frames=max_frames,
    )
    print("\n=== STEP 1: SAMPLING PLAN ===")
    print(f"Interval between samples: {interval_sec} sec")
    print(f"Frames to extract: {len(frame_indices)}")
    frame_records = extract_frames_with_timestamps(
        video_path=video_path, frame_indices=frame_indices, metadata=metadata, output_dir=output_dir,
    )
    manifest_path = save_step1_manifest(metadata, frame_records, output_dir=output_dir, manifest_path=manifest_path)
    print("\n=== STEP 1: RESULT ===")
    print(f"Frames extracted: {len(frame_records)}")
    print(f"Manifest: {manifest_path}")
    return {"metadata": asdict(metadata), "frame_records": frame_records, "manifest_path": manifest_path}


# --- Step 2: Face detection and tracking ---

def resolve_image_path(manifest_path: str, image_path: str) -> str:
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    if os.path.exists(image_path):
        return image_path
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    candidate = os.path.join(manifest_dir, os.path.basename(image_path))
    if os.path.exists(candidate):
        return candidate
    return image_path


def build_detector() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def detect_faces(
    img_bgr: np.ndarray,
    detector: cv2.CascadeClassifier,
    scale_factor: float = 1.05,
    min_neighbors: int = 6,
    min_size: Tuple[int, int] = (50, 50),
) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size,
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def _intersection_area(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    xi = max(xa, xb)
    yi = max(ya, yb)
    wi = max(0, min(xa + wa, xb + wb) - xi)
    hi = max(0, min(ya + ha, yb + hb) - yi)
    return float(wi * hi)


def drop_contained_detections(
    detections: List[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    """Remove detections that are mostly inside a larger one (e.g. eye inside face)."""
    if len(detections) < 2:
        return detections
    areas = [w * h for (_, _, w, h) in detections]
    keep = [True] * len(detections)
    for i in range(len(detections)):
        if not keep[i]:
            continue
        ai = areas[i]
        for j in range(len(detections)):
            if i == j or not keep[j]:
                continue
            aj = areas[j]
            inter = _intersection_area(detections[i], detections[j])
            if ai < aj and inter >= 0.5 * ai:
                keep[i] = False
                break
            if aj < ai and inter >= 0.5 * aj:
                keep[j] = False
    return [det for det, k in zip(detections, keep) if k]


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
    gray_vec = gray_small.flatten()
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_h = hist_h / (np.sum(hist_h) + 1e-6)
    hist_s = hist_s / (np.sum(hist_s) + 1e-6)
    emb = np.concatenate([gray_vec, hist_h, hist_s]).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-6)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def center_of(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def area_of(bbox: Tuple[int, int, int, int]) -> float:
    _, _, w, h = bbox
    return float(w * h)


def filter_face_detections(
    detections: List[Tuple[int, int, int, int]],
    frame_shape: Tuple[int, int, int],
    min_area_ratio: float = 0.001,
    aspect_ratio_min: float = 0.6,
    aspect_ratio_max: float = 1.8,
) -> List[Tuple[int, int, int, int]]:
    """Drop detections that are too small or have unface-like aspect ratio (e.g. eye/ear fragments)."""
    if not detections:
        return []
    h, w = frame_shape[:2]
    frame_area = max(1.0, float(w * h))
    min_area = max(50 * 50, frame_area * min_area_ratio)
    filtered = []
    for (x, y, bw, bh) in detections:
        area = bw * bh
        if area < min_area:
            continue
        aspect = bw / max(bh, 1)
        if aspect < aspect_ratio_min or aspect > aspect_ratio_max:
            continue
        filtered.append((x, y, bw, bh))
    return filtered


def match_score(
    det_embedding: np.ndarray,
    det_center: Tuple[float, float],
    det_area: float,
    track: TrackState,
    frame_diag: float,
) -> float:
    app_dist = cosine_distance(det_embedding, track.last_embedding)
    spatial_dist = np.linalg.norm(np.array(det_center) - np.array(track.last_center)) / max(frame_diag, 1.0)
    area_dist = abs(det_area - track.last_area) / max(det_area, track.last_area, 1.0)
    return 0.85 * app_dist + 0.10 * spatial_dist + 0.05 * area_dist


def assign_person_ids(
    detections: List[Tuple[int, int, int, int]],
    embeddings: List[np.ndarray],
    sample_index: int,
    frame_shape: Tuple[int, int, int],
    tracks: Dict[str, TrackState],
    next_person_num: int,
    max_inactive_frames: int = 8,
    threshold: float = 0.45,
    max_people: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    h, w = frame_shape[:2]
    frame_diag = float((h ** 2 + w ** 2) ** 0.5)

    if max_people == 2 and len(detections) <= 2:
        results, next_person_num = _assign_two_people_by_embedding(
            detections=detections,
            embeddings=embeddings,
            sample_index=sample_index,
            frame_shape=frame_shape,
            tracks=tracks,
            next_person_num=next_person_num,
            frame_diag=frame_diag,
            threshold=threshold,
        )
        return results, next_person_num

    active_tracks = {pid: tr for pid, tr in tracks.items() if sample_index - tr.last_sample_index <= max_inactive_frames}
    used_tracks = set()
    results: List[Dict[str, Any]] = []
    for det_idx, bbox in enumerate(detections):
        emb = embeddings[det_idx]
        det_center = center_of(bbox)
        det_area = area_of(bbox)
        best_pid, best_score = None, 10.0
        for pid, tr in active_tracks.items():
            if pid in used_tracks:
                continue
            score = match_score(emb, det_center, det_area, tr, frame_diag)
            if score < best_score:
                best_score, best_pid = score, pid
        if best_pid is None or best_score > threshold:
            if max_people is not None and len(tracks) >= max_people:
                best_pid, best_score = None, 10.0
                for pid, tr in tracks.items():
                    if pid in used_tracks:
                        continue
                    score = match_score(emb, det_center, det_area, tr, frame_diag)
                    if score < best_score:
                        best_score, best_pid = score, pid
                if best_pid is None:
                    best_pid = min(tracks.keys())
                    best_score = 0.0
            else:
                next_person_num += 1
                best_pid, best_score = f"P{next_person_num:03d}", 0.0
        used_tracks.add(best_pid)
        x, y, bw, bh = bbox
        tracks[best_pid] = TrackState(
            person_id=best_pid,
            last_embedding=emb,
            last_center=det_center,
            last_area=det_area,
            last_sample_index=sample_index,
            hits=tracks[best_pid].hits + 1 if best_pid in tracks else 1,
        )
        results.append({
            "face_idx": det_idx,
            "person_id": best_pid,
            "bbox": {"x": x, "y": y, "w": bw, "h": bh},
            "center": {"x": round(det_center[0], 2), "y": round(det_center[1], 2)},
            "area": int(det_area),
            "match_score": round(float(best_score), 4),
            "confidence": 1.0,
            "embedding_preview": [round(float(v), 5) for v in emb[:8]],
        })
    return results, next_person_num


def _assign_two_people_by_embedding(
    detections: List[Tuple[int, int, int, int]],
    embeddings: List[np.ndarray],
    sample_index: int,
    frame_shape: Tuple[int, int, int],
    tracks: Dict[str, TrackState],
    next_person_num: int,
    frame_diag: float,
    threshold: float,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Assign at most 2 detections to P001/P002 by best embedding+position match,
    so the same person keeps the same ID even when they swap left/right.
    """
    if not detections:
        return [], next_person_num
    p001, p002 = "P001", "P002"
    if next_person_num < 2:
        next_person_num = 2
    track_ids = [p001, p002]
    det_centers = [center_of(b) for b in detections]
    det_areas = [area_of(b) for b in detections]

    if len(detections) == 2 and len(tracks) >= 2:
        # Best pairing: assign each detection to the track it matches best (by appearance+position)
        scores = []
        for det_idx in range(2):
            row = []
            for pid in track_ids:
                if pid not in tracks:
                    row.append(10.0)
                    continue
                tr = tracks[pid]
                sc = match_score(
                    embeddings[det_idx], det_centers[det_idx], det_areas[det_idx], tr, frame_diag
                )
                row.append(sc)
            scores.append(row)
        # Greedy: pick best (det, track) pair, then assign the remainder
        if tracks.get(p001) and tracks.get(p002):
            (d0, t0), (d1, t1) = (0, p001), (1, p002)
            s00 = scores[0][0]
            s01 = scores[0][1]
            s10 = scores[1][0]
            s11 = scores[1][1]
            if s00 + s11 <= s01 + s10:
                assign = [(0, p001), (1, p002)]
            else:
                assign = [(0, p002), (1, p001)]
        else:
            order = sorted(range(len(detections)), key=lambda i: det_centers[i][0])
            assign = [(order[0], p001), (order[1], p002)]
    elif len(detections) == 2 and len(tracks) == 1:
        existing = p001 if p001 in tracks else p002
        other = p002 if existing == p001 else p001
        best_det_for_existing = 0 if match_score(
            embeddings[0], det_centers[0], det_areas[0], tracks[existing], frame_diag
        ) <= match_score(
            embeddings[1], det_centers[1], det_areas[1], tracks[existing], frame_diag
        ) else 1
        assign = [(best_det_for_existing, existing), (1 - best_det_for_existing, other)]
    elif len(detections) == 2:
        order = sorted(range(len(detections)), key=lambda i: det_centers[i][0])
        assign = [(order[0], p001), (order[1], p002)]
    elif len(detections) == 1:
        det_idx = 0
        emb, det_center, det_area = embeddings[0], det_centers[0], det_areas[0]
        if len(tracks) >= 2:
            best_pid, best_score = None, 10.0
            for pid in track_ids:
                if pid not in tracks:
                    continue
                sc = match_score(emb, det_center, det_area, tracks[pid], frame_diag)
                if sc < best_score:
                    best_score, best_pid = sc, pid
            assign = [(0, best_pid)] if best_pid else [(0, p001)]
        elif len(tracks) == 1:
            existing = p001 if p001 in tracks else p002
            sc = match_score(emb, det_center, det_area, tracks[existing], frame_diag)
            assign = [(0, existing)] if sc <= threshold else [(0, p002 if existing == p001 else p001)]
        else:
            assign = [(0, p001)]
    else:
        assign = []

    results = []
    for det_idx, pid in assign:
        bbox = detections[det_idx]
        emb = embeddings[det_idx]
        det_center = center_of(bbox)
        det_area = area_of(bbox)
        x, y, bw, bh = bbox
        tracks[pid] = TrackState(
            person_id=pid,
            last_embedding=emb,
            last_center=det_center,
            last_area=det_area,
            last_sample_index=sample_index,
            hits=tracks[pid].hits + 1 if pid in tracks else 1,
        )
        results.append({
            "face_idx": det_idx,
            "person_id": pid,
            "bbox": {"x": x, "y": y, "w": bw, "h": bh},
            "center": {"x": round(det_center[0], 2), "y": round(det_center[1], 2)},
            "area": int(det_area),
            "match_score": 0.0,
            "confidence": 1.0,
            "embedding_preview": [round(float(v), 5) for v in emb[:8]],
        })
    return results, next_person_num


def _assign_two_people_by_position(
    detections: List[Tuple[int, int, int, int]],
    embeddings: List[np.ndarray],
    sample_index: int,
    frame_shape: Tuple[int, int, int],
    tracks: Dict[str, TrackState],
    next_person_num: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Assign at most 2 detections by horizontal position: left → P001, right → P002."""
    if not detections:
        return [], next_person_num
    order = sorted(range(len(detections)), key=lambda i: center_of(detections[i])[0])
    p001, p002 = "P001", "P002"
    if next_person_num < 2:
        next_person_num = 2
    results = []
    for rank, det_idx in enumerate(order):
        bbox = detections[det_idx]
        emb = embeddings[det_idx]
        det_center = center_of(bbox)
        det_area = area_of(bbox)
        pid = p001 if rank == 0 else p002
        if len(detections) == 1 and len(tracks) >= 2:
            cx = det_center[0]
            best_pid, best_dx = None, float("inf")
            for tid in (p001, p002):
                if tid not in tracks:
                    continue
                dx = abs(tracks[tid].last_center[0] - cx)
                if dx < best_dx:
                    best_dx, best_pid = dx, tid
            if best_pid is not None:
                pid = best_pid
        x, y, bw, bh = bbox
        tracks[pid] = TrackState(
            person_id=pid,
            last_embedding=emb,
            last_center=det_center,
            last_area=det_area,
            last_sample_index=sample_index,
            hits=tracks[pid].hits + 1 if pid in tracks else 1,
        )
        results.append({
            "face_idx": det_idx,
            "person_id": pid,
            "bbox": {"x": x, "y": y, "w": bw, "h": bh},
            "center": {"x": round(det_center[0], 2), "y": round(det_center[1], 2)},
            "area": int(det_area),
            "match_score": 0.0,
            "confidence": 1.0,
            "embedding_preview": [round(float(v), 5) for v in emb[:8]],
        })
    return results, next_person_num


def attach_comments_to_frames(
    frame_records: List[Dict[str, Any]],
    comments: Optional[List[Dict[str, Any]]],
) -> None:
    if not comments or not frame_records:
        return
    frame_times = np.array([float(fr.get("timestamp_sec", 0.0)) for fr in frame_records], dtype=np.float32)
    for idx, comment in enumerate(comments):
        ts_sec = float(comment.get("timestamp_sec", parse_timestamp(comment["timestamp"]) if "timestamp" in comment else 0))
        nearest_idx = int(np.argmin(np.abs(frame_times - ts_sec)))
        target = frame_records[nearest_idx]
        target.setdefault("comments", [])
        target["comments"].append({
            "comment_id": comment.get("comment_id", f"C{idx+1:03d}"),
            "text": comment.get("text", ""),
            "timestamp_sec": round(ts_sec, 3),
            "timestamp": format_timestamp(ts_sec),
        })


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
            start_sec, end_sec = interval_id * interval_sec, (interval_id + 1) * interval_sec
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
        b = buckets[interval_id]
        b["frame_count"] += 1
        b["total_faces"] += int(fr.get("face_count", 0))
        for pid in fr.get("person_ids", []):
            b["persons"].add(pid)
            b["person_frame_hits"][pid] += 1
        if fr.get("comments"):
            b["comments"].extend(fr["comments"])
    out = []
    for interval_id in sorted(buckets.keys()):
        b = buckets[interval_id]
        fc = max(1, b["frame_count"])
        out.append({
            "interval_id": interval_id,
            "start_sec": b["start_sec"],
            "end_sec": b["end_sec"],
            "start_ts": b["start_ts"],
            "end_ts": b["end_ts"],
            "frame_count": b["frame_count"],
            "total_faces": b["total_faces"],
            "avg_faces_per_frame": round(b["total_faces"] / fc, 3),
            "persons": sorted(b["persons"]),
            "unique_person_count": len(b["persons"]),
            "person_presence_ratio": {pid: round(b["person_frame_hits"][pid] / fc, 3) for pid in sorted(b["persons"])},
            "comments": b["comments"],
        })
    return out


def compare_intervals(intervals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    comparisons = []
    for i in range(1, len(intervals)):
        prev_it, curr_it = intervals[i - 1], intervals[i]
        prev_set, curr_set = set(prev_it["persons"]), set(curr_it["persons"])
        union, intersection = prev_set | curr_set, prev_set & curr_set
        jaccard = (len(intersection) / len(union)) if union else 1.0
        comparisons.append({
            "from_interval_id": prev_it["interval_id"],
            "to_interval_id": curr_it["interval_id"],
            "from_ts": prev_it["start_ts"],
            "to_ts": curr_it["start_ts"],
            "persons_joined": sorted(curr_set - prev_set),
            "persons_left": sorted(prev_set - curr_set),
            "shared_persons": sorted(intersection),
            "person_jaccard": round(jaccard, 3),
            "avg_faces_delta": round(curr_it["avg_faces_per_frame"] - prev_it["avg_faces_per_frame"], 3),
            "face_count_delta": curr_it["total_faces"] - prev_it["total_faces"],
        })
    return comparisons


def summarize_persons(frame_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for fr in frame_records:
        ts, ts_txt = float(fr["timestamp_sec"]), fr["timestamp"]
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
            p["last_seen_sec"], p["last_seen_ts"] = ts, ts_txt
            if p["first_seen_sec"] > ts:
                p["first_seen_sec"], p["first_seen_ts"] = ts, ts_txt
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


def select_analysis_persons(
    person_summary: Dict[str, Dict[str, Any]],
    analysis_people: int,
) -> List[str]:
    """Select the top analysis_people by prominence (frames_seen * avg_area) for emotion/speaker analysis."""
    if not person_summary or analysis_people <= 0:
        return []
    if analysis_people >= len(person_summary):
        return sorted(person_summary.keys())
    prominence = [
        (pid, float(s.get("frames_seen", 0)) * float(s.get("avg_area", 0)))
        for pid, s in person_summary.items()
    ]
    prominence.sort(key=lambda x: -x[1])
    return [pid for pid, _ in prominence[:analysis_people]]


def _detection_feature(det: Dict[str, Any], width: int, height: int) -> np.ndarray:
    width, height = max(1, width), max(1, height)
    frame_area = max(1.0, float(width * height))
    center = det.get("center", {})
    cx = float(center.get("x", 0.0)) / float(width)
    cy = float(center.get("y", 0.0)) / float(height)
    area = max(1.0, float(det.get("area", 1.0)))
    log_area = float(np.log1p(area) / np.log1p(frame_area))
    emb = det.get("embedding_preview", [])[:8]
    emb = [float(v) for v in emb]
    emb.extend([0.0] * (8 - len(emb)))
    return np.array([cx, cy, log_area] + emb, dtype=np.float32)


def consolidate_people_kmeans(
    frame_records: List[Dict[str, Any]],
    video_metadata: Dict[str, Any],
    expected_people: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if expected_people <= 1:
        return frame_records, {"applied": False, "reason": "expected_people <= 1"}
    refs = [(fi, di, det) for fi, fr in enumerate(frame_records) for di, det in enumerate(fr.get("detections", []))]
    if len(refs) < 2:
        return frame_records, {"applied": False, "reason": "not enough detections"}
    width = int(video_metadata.get("width", 0) or 1)
    height = int(video_metadata.get("height", 0) or 1)
    features = np.vstack([_detection_feature(det, width, height) for _, _, det in refs]).astype(np.float32)
    k = min(expected_people, len(refs))
    if k < 2:
        return frame_records, {"applied": False, "reason": "k < 2"}
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, labels, _ = cv2.kmeans(features, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    label_values = [int(v) for v in labels.flatten().tolist()]
    raw_summary = summarize_persons(frame_records)
    dominant_raw_id = max(raw_summary.items(), key=lambda kv: kv[1]["frames_seen"])[0]
    label_counts = Counter(label_values)
    dominant_label_votes = Counter(
        lbl for (_, _, det), lbl in zip(refs, label_values) if det.get("person_id") == dominant_raw_id
    )
    dominant_label = dominant_label_votes.most_common(1)[0][0] if dominant_label_votes else label_counts.most_common(1)[0][0]
    ordered_labels = [dominant_label] + [lbl for lbl, _ in label_counts.most_common() if lbl != dominant_label]
    label_to_person = {lbl: f"P{idx + 1:03d}" for idx, lbl in enumerate(ordered_labels)}
    ref_label_map = {(fi, di): lbl for (fi, di, _), lbl in zip(refs, label_values)}
    raw_to_canonical_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (fi, di, det), lbl in zip(refs, label_values):
        raw_to_canonical_counts[str(det.get("person_id", "UNKNOWN"))][label_to_person[lbl]] += 1
    consolidated_frames = []
    for frame_idx, fr in enumerate(frame_records):
        best_by_label: Dict[int, Dict[str, Any]] = {}
        for det_idx, det in enumerate(fr.get("detections", [])):
            lbl = ref_label_map[(frame_idx, det_idx)]
            current = best_by_label.get(lbl)
            if current is None or float(det.get("area", 0.0)) > float(current.get("area", 0.0)):
                det_copy = dict(det)
                det_copy["source_person_id"] = det.get("person_id")
                det_copy["person_id"] = label_to_person[lbl]
                det_copy["cluster_label"] = int(lbl)
                best_by_label[lbl] = det_copy
        fr["detections"] = sorted(best_by_label.values(), key=lambda d: d["person_id"])
        fr["person_ids"] = sorted({d["person_id"] for d in fr["detections"]})
        fr["face_count"] = len(fr["detections"])
        consolidated_frames.append(fr)
    consolidated_summary = summarize_persons(consolidated_frames)
    info = {
        "applied": True,
        "method": "kmeans_detections",
        "expected_people": expected_people,
        "raw_unique_ids": len(raw_summary),
        "consolidated_unique_ids": len(consolidated_summary),
        "dominant_raw_id": dominant_raw_id,
        "label_to_person": {str(lbl): pid for lbl, pid in sorted(label_to_person.items())},
        "raw_to_canonical_votes": {r: dict(sorted(c.items())) for r, c in sorted(raw_to_canonical_counts.items())},
    }
    return consolidated_frames, info


def run_step2(
    manifest_path: str = "extracted_frames_v2/frame_manifest_step1.json",
    output_path: str = "extracted_frames_v2/analysis_step2.json",
    interval_sec: float = 10.0,
    comments: Optional[List[Dict[str, Any]]] = None,
    save_face_crops: bool = True,
    crops_dir: str = "extracted_frames_v2/faces_by_person",
    expected_people: Optional[int] = None,
    max_tracked_people: Optional[int] = None,
    analysis_people: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Detect faces, track persons, optionally consolidate clusters, and save face crops.
    Face crops are stored locally in crops_dir; canonical storage is Yandex Cloud
    (upload/sync of faces_by_person to bucket is done separately).
    """
    manifest = load_manifest(manifest_path)
    frame_records = manifest.get("frames", [])
    attach_comments_to_frames(frame_records, comments)
    detector = build_detector()
    if detector.empty():
        raise RuntimeError("Haar Cascade failed to load.")
    if save_face_crops:
        os.makedirs(crops_dir, exist_ok=True)
    tracks: Dict[str, TrackState] = {}
    next_person_num = 0
    max_people_for_tracking = max_tracked_people if max_tracked_people is not None else expected_people
    enriched_frames = []
    for fr in frame_records:
        sample_index = int(fr["sample_index"])
        img_path = resolve_image_path(manifest_path, fr["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            fr["detections"], fr["face_count"], fr["person_ids"] = [], 0, []
            enriched_frames.append(fr)
            continue
        detections = detect_faces(img, detector=detector)
        detections = drop_contained_detections(detections)
        detections = filter_face_detections(detections, img.shape)
        embeddings = [face_embedding(img, bbox) for bbox in detections]
        assigned, next_person_num = assign_person_ids(
            detections=detections, embeddings=embeddings, sample_index=sample_index,
            frame_shape=img.shape, tracks=tracks, next_person_num=next_person_num,
            max_people=max_people_for_tracking,
        )
        if save_face_crops:
            ih, iw = img.shape[:2]
            for det in assigned:
                pid = det["person_id"]
                x, y = det["bbox"]["x"], det["bbox"]["y"]
                bw, bh = det["bbox"]["w"], det["bbox"]["h"]
                x, y, bw, bh = clamp_bbox((x, y, bw, bh), iw, ih)
                crop = img[y : y + bh, x : x + bw]
                person_dir = os.path.join(crops_dir, pid)
                os.makedirs(person_dir, exist_ok=True)
                crop_name = f"s{sample_index:05d}_f{det['face_idx']:02d}.jpg"
                crop_path = os.path.join(person_dir, crop_name)
                cv2.imwrite(crop_path, crop)
                det["crop_path"] = crop_path
        fr["detections"], fr["face_count"] = assigned, len(assigned)
        fr["person_ids"] = sorted({d["person_id"] for d in assigned})
        enriched_frames.append(fr)
    consolidation_info = None
    if expected_people is not None and expected_people > 1:
        enriched_frames, consolidation_info = consolidate_people_kmeans(
            frame_records=enriched_frames,
            video_metadata=manifest.get("video_metadata", {}),
            expected_people=expected_people,
        )
    intervals = aggregate_intervals(enriched_frames, interval_sec=interval_sec)
    interval_comparison = compare_intervals(intervals)
    person_summary = summarize_persons(enriched_frames)
    if analysis_people is not None and analysis_people > 0:
        analysis_person_ids = select_analysis_persons(person_summary, analysis_people)
    else:
        analysis_person_ids = sorted(person_summary.keys())
    result = {
        "video_metadata": manifest.get("video_metadata", {}),
        "step2_config": {
            "interval_sec": interval_sec,
            "detector": "haar_frontalface_default",
            "tracking": "appearance+spatial matching",
            "comments_linked": bool(comments),
            "expected_people": expected_people,
            "max_tracked_people": max_tracked_people,
            "analysis_people": analysis_people,
            "analysis_person_ids": analysis_person_ids,
        },
        "frames": enriched_frames,
        "intervals": intervals,
        "interval_comparison": interval_comparison,
        "person_summary": person_summary,
        "analysis_person_ids": analysis_person_ids,
    }
    if consolidation_info is not None:
        result["consolidation"] = consolidation_info
    save_json(result, output_path)
    print("=== STEP 2: TRACKING + INTERVAL ANALYSIS ===")
    print(f"Frames processed: {len(enriched_frames)}")
    print(f"Total face detections: {sum(fr.get('face_count', 0) for fr in enriched_frames)}")
    print(f"Unique person IDs: {list(person_summary.keys())}")
    print(f"Saved: {output_path}")
    return result


# --- Step 3: Extended analytics ---

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
    enriched = []
    for fr in sorted(frames, key=lambda x: int(x.get("sample_index", 0))):
        ts_sec = float(fr.get("timestamp_sec", 0.0))
        fr["interval_id"] = int(ts_sec // interval_sec)
        for det in fr.get("detections", []):
            pid = str(det.get("person_id"))
            center = det.get("center", {})
            cx, cy = float(center.get("x", 0.0)), float(center.get("y", 0.0))
            area = float(det.get("area", 0.0))
            if area <= 0.0:
                bbox = det.get("bbox", {})
                area = float(bbox.get("w", 0.0)) * float(bbox.get("h", 0.0))
            area_ratio = area / frame_area
            area_log = float(np.log1p(max(area, 1.0)))
            motion_px = motion_px_per_sec = area_delta = area_ratio_delta = dt = 0.0
            if pid in previous_by_person:
                prev = previous_by_person[pid]
                prev_center, prev_area = prev["center"], float(prev["area"])
                prev_ts = float(prev["ts_sec"])
                dt = max(1.0 / fps, ts_sec - prev_ts)
                motion_px = _distance((cx, cy), prev_center)
                motion_px_per_sec = motion_px / dt
                area_delta = area - prev_area
                area_ratio_delta = area_delta / max(area, prev_area, 1.0)
            det["tracking_features"] = {
                "center_norm": {"x": round(cx / width, 6), "y": round(cy / height, 6)},
                "face_area_ratio": round(area_ratio, 6),
                "face_area_log": round(area_log, 6),
                "motion_px": round(motion_px, 4),
                "motion_px_per_sec": round(motion_px_per_sec, 4),
                "area_delta": round(area_delta, 2),
                "area_ratio_delta": round(area_ratio_delta, 6),
                "dt_sec": round(dt, 4),
            }
            det.setdefault("emotion", {"label": "unknown", "confidence": None, "model": "not_selected"})
            previous_by_person[pid] = {"center": (cx, cy), "area": area, "ts_sec": ts_sec}
        fr["face_count"] = len(fr.get("detections", []))
        fr["person_ids"] = sorted({str(d.get("person_id")) for d in fr.get("detections", []) if d.get("person_id")})
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
    linked = []
    for idx, comment in enumerate(comments):
        ts_sec = float(comment.get("timestamp_sec", parse_timestamp(comment["timestamp"]) if "timestamp" in comment else 0))
        nearest_idx = int(np.argmin(np.abs(frame_times - ts_sec)))
        frame = frames[nearest_idx]
        detections = frame.get("detections", [])
        assigned_person_id = max(detections, key=lambda d: float(d.get("area", 0.0))).get("person_id") if detections else None
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
                "person_stats": defaultdict(lambda: {"frames_seen": set(), "detections": 0, "area_ratio_sum": 0.0, "motion_sum": 0.0, "motion_max": 0.0, "emotion_counts": Counter()}),
            }
        b = buckets[interval_id]
        b["frame_count"] += 1
        if fr.get("comments"):
            b["comments"].extend(fr["comments"])
        for det in fr.get("detections", []):
            pid = str(det.get("person_id"))
            tf = det.get("tracking_features", {})
            emo = det.get("emotion", {}).get("label", "unknown")
            st = b["person_stats"][pid]
            st["frames_seen"].add(int(fr.get("sample_index", 0)))
            st["detections"] += 1
            st["area_ratio_sum"] += float(tf.get("face_area_ratio", 0.0))
            motion = float(tf.get("motion_px_per_sec", 0.0))
            st["motion_sum"] += motion
            st["motion_max"] = max(st["motion_max"], motion)
            st["emotion_counts"][emo] += 1
    out = []
    for interval_id in sorted(buckets.keys()):
        b = buckets[interval_id]
        fc = max(1, int(b["frame_count"]))
        person_stats_out = {}
        for pid, st in b["person_stats"].items():
            detections = max(1, int(st["detections"]))
            person_stats_out[pid] = {
                "frames_seen": len(st["frames_seen"]),
                "presence_ratio": round(len(st["frames_seen"]) / fc, 4),
                "detections": int(st["detections"]),
                "avg_face_area_ratio": round(st["area_ratio_sum"] / detections, 6),
                "avg_motion_px_per_sec": round(st["motion_sum"] / detections, 4),
                "max_motion_px_per_sec": round(st["motion_max"], 4),
                "emotion_counts": dict(st["emotion_counts"]),
            }
        persons = sorted(person_stats_out.keys())
        total_faces = int(sum(v["detections"] for v in person_stats_out.values()))
        out.append({
            "interval_id": interval_id,
            "start_sec": b["start_sec"],
            "end_sec": b["end_sec"],
            "start_ts": b["start_ts"],
            "end_ts": b["end_ts"],
            "frame_count": fc,
            "total_faces": total_faces,
            "avg_faces_per_frame": round(total_faces / fc, 4),
            "persons": persons,
            "unique_person_count": len(persons),
            "person_stats": person_stats_out,
            "comments": b["comments"],
        })
    return out


def compare_intervals_extended(intervals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    comparisons = []
    for i in range(1, len(intervals)):
        prev_it, curr_it = intervals[i - 1], intervals[i]
        prev_set, curr_set = set(prev_it["persons"]), set(curr_it["persons"])
        union, intersection = prev_set | curr_set, prev_set & curr_set
        jaccard = (len(intersection) / len(union)) if union else 1.0
        shared_deltas = []
        for pid in sorted(intersection):
            prev_stats = prev_it["person_stats"].get(pid, {})
            curr_stats = curr_it["person_stats"].get(pid, {})
            shared_deltas.append({
                "person_id": pid,
                "presence_ratio_delta": round(float(curr_stats.get("presence_ratio", 0.0)) - float(prev_stats.get("presence_ratio", 0.0)), 4),
                "avg_motion_delta": round(float(curr_stats.get("avg_motion_px_per_sec", 0.0)) - float(prev_stats.get("avg_motion_px_per_sec", 0.0)), 4),
                "avg_area_ratio_delta": round(float(curr_stats.get("avg_face_area_ratio", 0.0)) - float(prev_stats.get("avg_face_area_ratio", 0.0)), 6),
            })
        comparisons.append({
            "from_interval_id": prev_it["interval_id"],
            "to_interval_id": curr_it["interval_id"],
            "from_ts": prev_it["start_ts"],
            "to_ts": curr_it["start_ts"],
            "person_jaccard": round(jaccard, 4),
            "persons_joined": sorted(curr_set - prev_set),
            "persons_left": sorted(prev_set - curr_set),
            "shared_person_deltas": shared_deltas,
            "avg_faces_delta": round(float(curr_it.get("avg_faces_per_frame", 0.0)) - float(prev_it.get("avg_faces_per_frame", 0.0)), 4),
            "comment_count_delta": len(curr_it.get("comments", [])) - len(prev_it.get("comments", [])),
        })
    return comparisons


def build_person_timeline(
    frames: List[Dict[str, Any]],
    person_summary: Dict[str, Dict[str, Any]],
    interval_sec: float,
) -> Dict[str, Dict[str, Any]]:
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
    timeline = {}
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
    return {
        "candidates": [
            {"name": "fer (MiniXception, FER2013)", "package": "fer", "accuracy": "medium", "speed": "high", "cpu_friendly": True},
            {"name": "DeepFace (emotion backend)", "package": "deepface", "accuracy": "medium_high", "speed": "medium_low", "cpu_friendly": False},
            {"name": "HSEmotion (ONNX/PyTorch)", "package": "hsemotion", "accuracy": "high", "speed": "medium", "cpu_friendly": True},
        ],
        "recommendation": {"phase_1": {"model": "fer (MiniXception, FER2013)"}, "phase_2": {"model": "HSEmotion"}},
    }


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
    frames = enrich_detection_features(frames=frames, video_metadata=video_metadata, interval_sec=interval_sec)
    linked_comments = link_comments(frames, comments=comments)
    intervals_ext = build_extended_intervals(frames, interval_sec=interval_sec)
    interval_cmp_ext = compare_intervals_extended(intervals_ext)
    person_summary = result.get("person_summary", {})
    person_timeline = build_person_timeline(frames=frames, person_summary=person_summary, interval_sec=interval_sec)
    result["step3_config"] = {"interval_sec": interval_sec, "features": ["tracking_features", "interval_person_stats", "interval_comparison_extended", "comment_to_person_linking"]}
    result["frames"] = frames
    result["intervals_extended"] = intervals_ext
    result["interval_comparison_extended"] = interval_cmp_ext
    result["linked_comments"] = linked_comments
    result["person_timeline"] = person_timeline
    result["emotion_model_review"] = emotion_model_review()
    save_json(result, output_path)
    print("=== STEP 3: EXTENDED ANALYTICS ===")
    print(f"Frames: {len(frames)}")
    print(f"Saved: {output_path}")
    return result


# --- Step 4: Balanced sampling (dense frames for emotion) ---

def build_balanced_sampling_plan(
    metadata: VideoMetadata,
    interval_sec: float = 10.0,
    frames_per_interval: int = 4,
    max_frames: Optional[int] = None,
) -> List[int]:
    if metadata.frame_count <= 0:
        return []
    if metadata.duration_sec is None:
        metadata.duration_sec = metadata.frame_count / max(metadata.fps, 1.0)
    interval_count = max(1, int(math.ceil(metadata.duration_sec / interval_sec)))
    frame_indices = []
    for interval_id in range(interval_count):
        start_sec = interval_id * interval_sec
        end_sec = min(metadata.duration_sec, (interval_id + 1) * interval_sec)
        if end_sec <= start_sec:
            continue
        for sample_idx in range(frames_per_interval):
            ratio = (sample_idx + 1) / (frames_per_interval + 1)
            ts_sec = start_sec + ratio * (end_sec - start_sec)
            frame_idx = int(round(ts_sec * metadata.fps))
            frame_idx = min(max(0, frame_idx), max(0, metadata.frame_count - 1))
            frame_indices.append(frame_idx)
    frame_indices = sorted(set(frame_indices))
    if max_frames is not None and max_frames > 0 and len(frame_indices) > max_frames:
        positions = [int(round(i * (len(frame_indices) - 1) / (max_frames - 1))) for i in range(max_frames)]
        frame_indices = [frame_indices[p] for p in positions]
    return frame_indices


def extract_frames(
    video_path: str,
    frame_indices: List[int],
    fps: float,
    output_dir: str,
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    records = []
    for sample_index, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ts_sec = frame_idx / max(fps, 1.0)
        file_name = f"frame_{sample_index:05d}_f{frame_idx:06d}.jpg"
        file_path = os.path.join(output_dir, file_name)
        cv2.imwrite(file_path, frame)
        records.append({
            "sample_index": sample_index,
            "frame_index": int(frame_idx),
            "timestamp_sec": round(ts_sec, 6),
            "timestamp": format_timestamp(ts_sec),
            "image_path": file_path,
            "interval_id": None,
            "person_ids": [],
            "comments": [],
        })
    cap.release()
    return records


def save_manifest(
    metadata: VideoMetadata,
    frame_records: List[Dict[str, Any]],
    interval_sec: float,
    frames_per_interval: int,
    output_dir: str,
    file_name: str = "frame_manifest_step4_dense.json",
    manifest_path: Optional[str] = None,
) -> str:
    manifest = {
        "video_metadata": asdict(metadata),
        "sampling": {
            "strategy": "balanced_interval_sampling",
            "interval_sec": interval_sec,
            "frames_per_interval": frames_per_interval,
            "total_samples": len(frame_records),
            "fps_based": True,
        },
        "frames": frame_records,
    }
    path = manifest_path or os.path.join(output_dir, file_name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def run_step4(
    video_path: str = "video2.webm",
    output_dir: str = "extracted_frames_v3_dense",
    interval_sec: float = 10.0,
    frames_per_interval: int = 6,
    max_frames: Optional[int] = 1200,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    metadata = read_video_metadata(video_path)
    frame_indices = build_balanced_sampling_plan(
        metadata=metadata,
        interval_sec=interval_sec,
        frames_per_interval=frames_per_interval,
        max_frames=max_frames,
    )
    frame_records = extract_frames(video_path=video_path, frame_indices=frame_indices, fps=metadata.fps, output_dir=output_dir)
    manifest_path = save_manifest(metadata=metadata, frame_records=frame_records, interval_sec=interval_sec, frames_per_interval=frames_per_interval, output_dir=output_dir, manifest_path=manifest_path)
    print("=== STEP 4: DENSE + BALANCED SAMPLING ===")
    print(f"Video: {video_path}")
    print(f"Total frames: {len(frame_records)}")
    print(f"Manifest: {manifest_path}")
    return {"metadata": asdict(metadata), "frame_records": frame_records, "manifest_path": manifest_path}


if __name__ == "__main__":
    run_step1()
