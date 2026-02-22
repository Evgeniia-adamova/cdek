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

        records.append(
            {
                "sample_index": sample_idx,
                "frame_index": frame_idx,
                "timestamp_sec": round(timestamp_sec, 6),
                "timestamp": format_timestamp(timestamp_sec),
                "image_path": file_path,
                # Fields for subsequent analysis steps:
                "interval_id": None,
                "person_ids": [],
                "comments": [],
            }
        )

    cap.release()
    return records


def save_step1_manifest(
    metadata: VideoMetadata,
    frame_records: List[Dict[str, Any]],
    output_dir: str = "extracted_frames_v2",
) -> str:
    manifest = {
        "video_metadata": asdict(metadata),
        "sampling": {
            "total_samples": len(frame_records),
            "fps_based": True,
        },
        "frames": frame_records,
    }

    manifest_path = os.path.join(output_dir, "frame_manifest_step1.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def run_step1(
    video_path: str = "video2.webm",
    interval_sec: float = 0.5,
    max_frames: Optional[int] = 200,
    output_dir: str = "extracted_frames_v2",
) -> Dict[str, Any]:
    # Step 1: separate stage for reading FPS and metadata.
    metadata = read_video_metadata(video_path)
    print("=== STEP 1: VIDEO METADATA ===")
    print(f"Video: {metadata.video_path}")
    print(f"FPS (from metadata): {metadata.fps:.3f}")
    print(f"Total frames: {metadata.frame_count}")
    print(f"Resolution: {metadata.width}x{metadata.height}")
    if metadata.duration_sec is not None:
        print(f"Duration: {metadata.duration_sec:.2f} sec")

    frame_indices = build_sampling_plan(
        metadata=metadata,
        interval_sec=interval_sec,
        start_sec=0.0,
        end_sec=None,
        max_frames=max_frames,
    )

    print("\n=== STEP 1: SAMPLING PLAN ===")
    print(f"Sample interval: {interval_sec} sec")
    print(f"Step in frames (calculated via FPS): {max(1, int(round(interval_sec * metadata.fps)))}")
    print(f"Frames to extract: {len(frame_indices)}")

    frame_records = extract_frames_with_timestamps(
        video_path=video_path,
        frame_indices=frame_indices,
        metadata=metadata,
        output_dir=output_dir,
    )
    manifest_path = save_step1_manifest(metadata, frame_records, output_dir=output_dir)

    print("\n=== STEP 1: RESULT ===")
    print(f"Frames extracted: {len(frame_records)}")
    print(f"Manifest: {manifest_path}")
    if frame_records:
        print("Sample record:")
        print(frame_records[0])

    return {
        "metadata": asdict(metadata),
        "frame_records": frame_records,
        "manifest_path": manifest_path,
    }


if __name__ == "__main__":
    run_step1()
