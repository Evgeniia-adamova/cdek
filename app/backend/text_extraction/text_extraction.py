"""
Text extraction module: transcribe audio to text (Whisper).
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
        lines = [f"[{_format_ts(s['start'])} - {_format_ts(s['end'])}] {s['text']}" for s in segments]
        out_ts_txt.write_text("\n".join(lines), encoding="utf-8")

    return {
        "text": text,
        "language": result.get("language", language),
        "output_path": str(out_txt),
        "segments": segments,
    }


def main() -> int:
    script_dir = Path(__file__).resolve().parent.parent.parent.parent  # project root (app/backend/text_extraction -> root)
    default_audio = script_dir / "data" / "video_from_bucket_audio.ogg"

    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
    else:
        audio_path = default_audio

    if not audio_path.is_file():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        print("Usage: python -m app.backend.text_extraction.text_extraction [path_to_audio]", file=sys.stderr)
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
        print("Segments (JSON):", seg_path)
    print("Text length:", len(result["text"]), "chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
