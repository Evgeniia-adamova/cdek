#!/usr/bin/env python
# coding: utf-8
"""
Transcribe audio to text using OpenAI Whisper (runs locally, no API key).
Use this as an alternative to Yandex SpeechKit for speech-to-text.

Usage:
    pip install openai-whisper
    python transcribe_audio_whisper.py [path_to_audio]
    Default audio: data/video_from_bucket_audio.ogg

Output:
    Saves text to <audio_stem>.txt in the same folder as the audio (e.g. data/video_from_bucket_audio.txt).
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union


def transcribe(
    audio_path: Union[str, Path],
    model_name: str = "base",
    language: str = "ru",
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Transcribe audio file to text using Whisper.

    Args:
        audio_path: Path to audio file (.ogg, .wav, .mp3, etc.).
        model_name: Whisper model: "tiny", "base", "small", "medium", "large".
        language: Language code (e.g. "ru" for Russian).
        output_path: Where to save .txt. If None, uses <audio_stem>.txt next to audio.
        verbose: Whether to print progress.

    Returns:
        Dict with "text", "language", "output_path", "segments" (if needed).
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Install Whisper: pip install openai-whisper")

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    model = whisper.load_model(model_name)
    result = model.transcribe(str(path), language=language, verbose=verbose)

    text = (result.get("text") or "").strip()
    out = output_path or path.with_suffix(".txt")
    out = Path(out)
    out.write_text(text, encoding="utf-8")

    return {
        "text": text,
        "language": result.get("language", language),
        "output_path": str(out),
        "segments": result.get("segments", []),
    }


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_audio = script_dir / "data" / "video_from_bucket_audio.ogg"

    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
    else:
        audio_path = default_audio

    if not audio_path.is_file():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        print("Usage: python transcribe_audio_whisper.py [path_to_audio]", file=sys.stderr)
        return 1

    print("Loading Whisper model (base)...")
    result = transcribe(
        audio_path,
        model_name="base",
        language="ru",
        output_path=audio_path.with_suffix(".txt"),
        verbose=True,
    )
    print("\n=== Transcription ===")
    print("Output file:", result["output_path"])
    print("Text length:", len(result["text"]), "chars")
    print("\n--- Text ---")
    print(result["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
