"""
Unified speech analysis module combining:
- SpeechKit for speech-to-text
- Whisper/Torch for speaker diarization
- Hugging Face for sentiment analysis
"""
from app.backend.unified_speech_analysis.unified_analysis import (
    UnifiedSpeechAnalyzer,
    run_unified_analysis,
    main,
)

__all__ = [
    "UnifiedSpeechAnalyzer",
    "run_unified_analysis",
    "main",
]
