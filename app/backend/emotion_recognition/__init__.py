"""
Emotion recognition module: ONNX FER+ inference, report, and visualization.
"""

from app.backend.emotion_recognition.emotion_recognition import (
    run_emotion_baseline_onnx,
    run_step6_emotion_report,
    run_emotion_visual,
    plot_emotion_report,
)

__all__ = [
    "run_emotion_baseline_onnx",
    "run_step6_emotion_report",
    "run_emotion_visual",
    "plot_emotion_report",
]
