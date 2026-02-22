#!/usr/bin/env python
# coding: utf-8
"""
Run the full video pipeline (frame extraction → emotion report) using the video
stored in the Yandex bucket. Downloads the video to data/video_from_bucket.<ext>
then runs steps 1–6.

Requires .env with YANDEX_S3_BUCKET, YANDEX_S3_ACCESS_KEY_ID, YANDEX_S3_SECRET_ACCESS_KEY.
Optional: YANDEX_VIDEO_KEY (e.g. video_files/xxx.webm); otherwise first video under
video_files/ is used.

For emotion (step 5): pip install onnxruntime opencv-python

Usage:
    python run_pipeline_from_bucket.py
"""

import os
import sys
from pathlib import Path

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)


def main():
    from speech_processor import get_video_path_from_bucket

    print("=== DOWNLOADING VIDEO FROM YANDEX BUCKET ===")
    video_path = get_video_path_from_bucket(local_dir=str(SCRIPT_DIR / "data"))
    print(f"Using video: {video_path}\n")

    # Step 1: extract frames (v2 sampling)
    from video_pipeline_step1 import run_step1

    step1_out = run_step1(
        video_path=video_path,
        interval_sec=0.5,
        max_frames=200,
        output_dir="extracted_frames_v2",
    )
    manifest_v2 = step1_out["manifest_path"]

    # Step 2 (v2): face detection + tracking, output consolidated for step3
    from video_pipeline_step2 import run_step2

    run_step2(
        manifest_path=manifest_v2,
        output_path="extracted_frames_v2/analysis_step2_consolidated.json",
        interval_sec=10.0,
        save_face_crops=True,
        crops_dir="extracted_frames_v2/faces_by_person",
        expected_people=2,
    )

    # Step 3 (v2): extended analytics
    from video_pipeline_step3 import run_step3

    run_step3(
        input_path="extracted_frames_v2/analysis_step2_consolidated.json",
        output_path="extracted_frames_v2/analysis_step3_extended.json",
        interval_sec=10.0,
    )

    # Step 4: dense balanced sampling (same video)
    from video_pipeline_step4 import run_step4

    step4_out = run_step4(
        video_path=video_path,
        output_dir="extracted_frames_v3_dense",
        interval_sec=10.0,
        frames_per_interval=6,
        max_frames=1200,
    )
    manifest_v3 = step4_out["manifest_path"]

    # Step 2 (v3_dense): face detection on dense frames for emotion
    run_step2(
        manifest_path=manifest_v3,
        output_path="extracted_frames_v3_dense/analysis_step2_dense_consolidated.json",
        interval_sec=10.0,
        save_face_crops=True,
        crops_dir="extracted_frames_v3_dense/faces_by_person",
        expected_people=2,
    )

    # Step 3 (v3_dense): extended analytics for emotion pipeline
    run_step3(
        input_path="extracted_frames_v3_dense/analysis_step2_dense_consolidated.json",
        output_path="extracted_frames_v3_dense/analysis_step3_extended.json",
        interval_sec=10.0,
    )

    # Step 5: emotion baseline (ONNX FER+)
    from video_pipeline_step5_emotion_onnx import run_emotion_baseline_onnx

    run_emotion_baseline_onnx(
        analysis_path="extracted_frames_v3_dense/analysis_step2_dense_consolidated.json",
        output_path="extracted_frames_v3_dense/emotion_baseline_onnx_ferplus.json",
        max_per_person=300,
    )

    # Step 6: emotion report
    from video_pipeline_step6_emotion_report import run_step6_emotion_report

    run_step6_emotion_report(
        analysis_path="extracted_frames_v3_dense/analysis_step3_extended.json",
        emotion_path="extracted_frames_v3_dense/emotion_baseline_onnx_ferplus.json",
        output_path="extracted_frames_v3_dense/analysis_step6_emotion_report.json",
    )

    print("\n=== PIPELINE COMPLETE ===")
    print("Outputs:")
    print("  - extracted_frames_v2/ (metadata, frames, step2/step3)")
    print("  - extracted_frames_v3_dense/ (dense frames, emotion, report)")
    print("  - extracted_frames_v3_dense/analysis_step6_emotion_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
