"""
Face detection module: frame extraction, face detection, tracking, enrichment.
Merged from pipeline steps 1–4 (metadata, sampling, detection, consolidation, extended analytics).
"""

from app.backend.face_detection.face_detection import (
    run_step1,
    run_step2,
    run_step3,
    run_step4,
    read_video_metadata,
    build_sampling_plan,
    build_balanced_sampling_plan,
)

__all__ = [
    "run_step1",
    "run_step2",
    "run_step3",
    "run_step4",
    "read_video_metadata",
    "build_sampling_plan",
    "build_balanced_sampling_plan",
]
