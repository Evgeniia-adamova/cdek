import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import cv2


@dataclass
class VideoMetadata:
    video_path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: Optional[float]


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


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
    frame_indices: List[int] = []

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

    # Remove duplicates that sometimes appear on a short final interval.
    frame_indices = sorted(set(frame_indices))

    if max_frames is not None and max_frames > 0 and len(frame_indices) > max_frames:
        positions = [
            int(round(i * (len(frame_indices) - 1) / (max_frames - 1)))
            for i in range(max_frames)
        ]
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

    records: List[Dict[str, Any]] = []
    for sample_index, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        ts_sec = frame_idx / max(fps, 1.0)
        file_name = f"frame_{sample_index:05d}_f{frame_idx:06d}.jpg"
        file_path = os.path.join(output_dir, file_name)
        cv2.imwrite(file_path, frame)

        records.append(
            {
                "sample_index": sample_index,
                "frame_index": int(frame_idx),
                "timestamp_sec": round(ts_sec, 6),
                "timestamp": format_timestamp(ts_sec),
                "image_path": file_path,
                "interval_id": None,
                "person_ids": [],
                "comments": [],
            }
        )

    cap.release()
    return records


def save_manifest(
    metadata: VideoMetadata,
    frame_records: List[Dict[str, Any]],
    interval_sec: float,
    frames_per_interval: int,
    output_dir: str,
    file_name: str = "frame_manifest_step4_dense.json",
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

    manifest_path = os.path.join(output_dir, file_name)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def run_step4(
    video_path: str = "video2.webm",
    output_dir: str = "extracted_frames_v3_dense",
    interval_sec: float = 10.0,
    frames_per_interval: int = 6,
    max_frames: Optional[int] = 1200,
) -> Dict[str, Any]:
    metadata = read_video_metadata(video_path)
    frame_indices = build_balanced_sampling_plan(
        metadata=metadata,
        interval_sec=interval_sec,
        frames_per_interval=frames_per_interval,
        max_frames=max_frames,
    )

    frame_records = extract_frames(
        video_path=video_path,
        frame_indices=frame_indices,
        fps=metadata.fps,
        output_dir=output_dir,
    )
    manifest_path = save_manifest(
        metadata=metadata,
        frame_records=frame_records,
        interval_sec=interval_sec,
        frames_per_interval=frames_per_interval,
        output_dir=output_dir,
    )

    print("=== STEP 4: DENSE + BALANCED SAMPLING ===")
    print(f"Video: {video_path}")
    print(f"FPS: {metadata.fps:.3f}")
    print(f"Duration: {metadata.duration_sec:.2f} sec")
    print(f"Interval for balance: {interval_sec} sec")
    print(f"Frames per interval: {frames_per_interval}")
    print(f"Total frames: {len(frame_records)}")
    if frame_records:
        print(
            "Time coverage: "
            f"{frame_records[0]['timestamp']} -> {frame_records[-1]['timestamp']}"
        )
    print(f"Manifest: {manifest_path}")

    return {
        "metadata": asdict(metadata),
        "frame_records": frame_records,
        "manifest_path": manifest_path,
    }


if __name__ == "__main__":
    run_step4()
