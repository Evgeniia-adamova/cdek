#!/usr/bin/env python
# coding: utf-8
"""
Transcribe audio to text using OpenAI Whisper (runs locally, no API key).
Outputs plain text and timestamped segments (JSON + readable .timestamped.txt).

Usage:
    pip install openai-whisper
    python transcribe_audio_whisper.py [path_to_audio]
    Default audio: data/video_from_bucket_audio.ogg

Output:
    - <stem>.txt          : full text only
    - <stem>_segments.json: segments with start, end, text (for diarization/sentiment)
    - <stem>.timestamped.txt: human-readable [HH:MM:SS.mmm - HH:MM:SS.mmm] text
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _format_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def transcribe(
    audio_path: Union[str, Path],
    model_name: str = "base",
    language: str = "ru",
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    save_timestamps: bool = True,
) -> Dict[str, Any]:
    """
    Transcribe audio file to text using Whisper. Saves full text and timestamped segments.

    Returns:
        Dict with "text", "language", "output_path", "segments" (list of {start, end, text}).
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
    segments_raw = result.get("segments") or []
    segments: List[Dict[str, Any]] = [
        {"start": s["start"], "end": s["end"], "text": (s.get("text") or "").strip()}
        for s in segments_raw
        if (s.get("text") or "").strip()
    ]

    base = path.parent / path.stem
    out_txt = output_path or (base.with_suffix(".txt"))
    out_txt = Path(out_txt)
    out_txt.write_text(text, encoding="utf-8")

    if save_timestamps and segments:
        out_json = base.parent / f"{base.name}_segments.json"
        out_json.write_text(
            json.dumps({"language": result.get("language", language), "segments": segments}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        out_ts_txt = base.parent / f"{base.name}.timestamped.txt"
        lines = []
        for s in segments:
            line = f"[{_format_ts(s['start'])} - {_format_ts(s['end'])}] {s['text']}"
            lines.append(line)
        out_ts_txt.write_text("\n".join(lines), encoding="utf-8")

    return {
        "text": text,
        "language": result.get("language", language),
        "output_path": str(out_txt),
        "segments": segments,
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
    if result.get("segments"):
        seg_path = audio_path.parent / f"{audio_path.stem}_segments.json"
        ts_path = audio_path.parent / f"{audio_path.stem}.timestamped.txt"
        print("Segments (JSON):", seg_path)
        print("Timestamped txt:", ts_path)
    print("Text length:", len(result["text"]), "chars")
    print("\n--- Text ---")
    print(result["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
