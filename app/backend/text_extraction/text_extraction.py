"""
Text extraction module: transcribe audio to text using Yandex SpeechKit async API.

Uploads the full audio to S3, runs async (long-running) recognition on the entire
file at once, and returns a complete transcript with per-utterance timestamps.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _format_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _run_ffmpeg(args: list) -> None:
    """Run ffmpeg with given arguments. Raises on failure."""
    cmd = ["ffmpeg", "-y"] + args
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (code {completed.returncode}): "
            f"{completed.stderr.decode(errors='ignore')[-500:]}"
        )


def _normalize_to_opus(input_path: Path, output_path: Path) -> None:
    """Normalize audio to mono OGG/Opus at 48kHz (required for SpeechKit async API)."""
    _run_ffmpeg([
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "48000",
        "-c:a", "libopus",
        str(output_path),
    ])


def transcribe(
    audio_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    language: str = "ru-RU",
    verbose: bool = True,
    save_timestamps: bool = True,
) -> Dict[str, Any]:
    """
    Transcribe audio file using Yandex SpeechKit async (long-running) API.

    Uploads the full audio to S3, submits an async recognition request,
    polls until completion, and returns the complete transcript with timestamps.

    Args:
        audio_path: Path to audio file (any format ffmpeg supports).
        output_path: Path for output transcript .txt file.
        language: Language code for SpeechKit (default: ru-RU).
        verbose: Print progress messages.
        save_timestamps: Save segments JSON and timestamped txt alongside transcript.

    Returns:
        Dict with "text", "language", "output_path", "segments".
    """
    from app.backend.speech_processor import (
        SpeechProcessor,
        create_yandex_s3_client_from_env,
        ENV_YANDEX_S3_BUCKET,
        ENV_YANDEX_FOLDER_ID,
        ENV_YANDEX_API_KEY,
    )

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Build SpeechProcessor
    folder_id = os.environ.get(ENV_YANDEX_FOLDER_ID, "")
    api_key = os.environ.get(ENV_YANDEX_API_KEY, "")
    if not api_key:
        raise EnvironmentError(
            f"{ENV_YANDEX_API_KEY} is not set. Configure it in .env"
        )

    s3_client = create_yandex_s3_client_from_env()
    bucket = os.environ.get(ENV_YANDEX_S3_BUCKET, "")

    processor = SpeechProcessor(
        folder_id=folder_id,
        api_key=api_key,
        s3_client=s3_client,
        bucket=bucket,
    )

    # 1) Normalize to mono Opus (SpeechKit async requires OGG_OPUS)
    normalized = path.parent / f"{path.stem}_mono.ogg"
    if verbose:
        print("Normalizing audio to mono OGG/Opus...")
    _normalize_to_opus(path, normalized)

    try:
        # 2) Upload the full audio to S3
        if verbose:
            print("Uploading audio to S3...")
        audio_bytes = normalized.read_bytes()
        upload_result = processor.upload_voice_file(audio_bytes, file_extension=".ogg", public=True)

        if upload_result.get("status") != "uploaded":
            raise RuntimeError(
                f"S3 upload failed: {upload_result.get('error', 'unknown')}"
            )

        s3_key = upload_result["s3_key"]
        if verbose:
            print(f"Uploaded to S3: {s3_key}")

        # 3) Run async recognition on the full file
        if verbose:
            print("Running async SpeechKit recognition (this may take a few minutes)...")

        result = processor.async_speech_to_text(
            s3_key=s3_key,
            language_code=language,
        )
    finally:
        # Clean up normalized temp file
        if normalized.is_file() and normalized != path:
            normalized.unlink(missing_ok=True)

    if result.get("status") != "success":
        raise RuntimeError(
            f"Async STT failed: {result.get('error', result.get('status'))}"
        )

    full_text = result.get("text", "")
    segments: List[Dict[str, Any]] = result.get("segments", [])

    if verbose:
        print(f"Transcription complete: {len(full_text)} chars, {len(segments)} segments")

    # 4) Save outputs
    base = path.parent / path.stem
    out_txt = Path(output_path) if output_path else base.with_suffix(".txt")
    out_txt.write_text(full_text, encoding="utf-8")

    if save_timestamps and segments:
        out_json = base.parent / f"{base.name}_segments.json"
        out_json.write_text(
            json.dumps(
                {"language": language, "segments": segments},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        out_ts_txt = base.parent / f"{base.name}.timestamped.txt"
        lines = [
            f"[{_format_ts(s['start'])} - {_format_ts(s['end'])}] {s['text']}"
            for s in segments
        ]
        out_ts_txt.write_text("\n".join(lines), encoding="utf-8")

    return {
        "text": full_text,
        "language": language,
        "output_path": str(out_txt),
        "segments": segments,
    }


def main() -> int:
    """CLI entry point."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if len(sys.argv) < 2:
        print("Usage: python -m app.backend.text_extraction [path_to_audio]", file=sys.stderr)
        return 1

    audio_path = Path(sys.argv[1])
    if not audio_path.is_file():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        return 1

    print("Running async SpeechKit STT...")
    result = transcribe(
        audio_path,
        output_path=audio_path.with_suffix(".txt"),
        verbose=True,
    )
    print("\n=== Transcription ===")
    print("Output file:", result["output_path"])
    print("Text length:", len(result["text"]), "chars")
    print("Segments:", len(result["segments"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
