"""
Text analysis module: diarization + sentiment on transcript segments.
"""
from text_analysis.text_analysis import (
    load_segments,
    diarize_audio,
    assign_speaker_to_segment,
    run_sentiment_ru,
    run_diarize_sentiment,
    main,
)

__all__ = [
    "load_segments",
    "diarize_audio",
    "assign_speaker_to_segment",
    "run_sentiment_ru",
    "run_diarize_sentiment",
    "main",
]
