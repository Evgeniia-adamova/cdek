#!/usr/bin/env python
# coding: utf-8
"""
Run the full video pipeline (frame extraction → emotion report) using the video
stored in the Yandex bucket. Downloads the video to data/video_from_bucket.<ext>
then runs face_detection and emotion_recognition modules.

Outputs:
  - logs/ : extracted frame images and face crops (extracted_frames_v2, extracted_frames_v3_dense)
  - data/ : all JSON (manifests, analysis, emotion reports) in data/v2/ and data/v3_dense/

Requires .env with YANDEX_S3_BUCKET, YANDEX_S3_ACCESS_KEY_ID, YANDEX_S3_SECRET_ACCESS_KEY.
Optional: YANDEX_VIDEO_KEY (e.g. video_files/xxx.webm).

Usage (from project root):
    python -m app.run_pipeline_from_bucket
    or: python app/run_pipeline_from_bucket.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Paths: frames/crops → logs/, JSON → data/
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
FRAMES_V2 = str(LOGS_DIR / "extracted_frames_v2")
FRAMES_V3 = str(LOGS_DIR / "extracted_frames_v3_dense")
DATA_V2 = str(DATA_DIR / "v2")
DATA_V3 = str(DATA_DIR / "v3_dense")


def main():
    from app.speech_processor import get_video_path_from_bucket
    from app.backend.face_detection import run_step1, run_step2, run_step3, run_step4
    from app.backend.emotion_recognition import run_emotion_baseline_onnx, run_step6_emotion_report

    os.makedirs(DATA_V2, exist_ok=True)
    os.makedirs(DATA_V3, exist_ok=True)

    print("=== DOWNLOADING VIDEO FROM YANDEX BUCKET ===")
    video_path = get_video_path_from_bucket(local_dir=str(DATA_DIR))
    print(f"Using video: {video_path}\n")

    step1_out = run_step1(
        video_path=video_path,
        interval_sec=0.5,
        max_frames=200,
        output_dir=FRAMES_V2,
        manifest_path=os.path.join(DATA_V2, "frame_manifest_step1.json"),
    )
    manifest_v2 = step1_out["manifest_path"]

    run_step2(
        manifest_path=manifest_v2,
        output_path=os.path.join(DATA_V2, "analysis_step2_consolidated.json"),
        interval_sec=10.0,
        save_face_crops=True,
        crops_dir=os.path.join(FRAMES_V2, "faces_by_person"),
        expected_people=2,
    )

    run_step3(
        input_path=os.path.join(DATA_V2, "analysis_step2_consolidated.json"),
        output_path=os.path.join(DATA_V2, "analysis_step3_extended.json"),
        interval_sec=10.0,
    )

    step4_out = run_step4(
        video_path=video_path,
        output_dir=FRAMES_V3,
        interval_sec=10.0,
        frames_per_interval=6,
        max_frames=1200,
        manifest_path=os.path.join(DATA_V3, "frame_manifest_step4_dense.json"),
    )
    manifest_v3 = step4_out["manifest_path"]

    run_step2(
        manifest_path=manifest_v3,
        output_path=os.path.join(DATA_V3, "analysis_step2_dense_consolidated.json"),
        interval_sec=10.0,
        save_face_crops=True,
        crops_dir=os.path.join(FRAMES_V3, "faces_by_person"),
        expected_people=2,
    )

    run_step3(
        input_path=os.path.join(DATA_V3, "analysis_step2_dense_consolidated.json"),
        output_path=os.path.join(DATA_V3, "analysis_step3_extended.json"),
        interval_sec=10.0,
    )

    run_emotion_baseline_onnx(
        analysis_path=os.path.join(DATA_V3, "analysis_step2_dense_consolidated.json"),
        output_path=os.path.join(DATA_V3, "emotion_baseline_onnx_ferplus.json"),
        max_per_person=300,
    )

    run_step6_emotion_report(
        analysis_path=os.path.join(DATA_V3, "analysis_step3_extended.json"),
        emotion_path=os.path.join(DATA_V3, "emotion_baseline_onnx_ferplus.json"),
        output_path=os.path.join(DATA_V3, "analysis_step6_emotion_report.json"),
    )

    print("\n=== PIPELINE COMPLETE ===")
    print("Outputs:")
    print("  - logs/extracted_frames_v2/     (frame images, faces_by_person)")
    print("  - logs/extracted_frames_v3_dense/ (frame images, faces_by_person)")
    print("  - data/v2/                     (manifests, analysis JSON)")
    print("  - data/v3_dense/                (manifests, analysis, emotion JSON)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
