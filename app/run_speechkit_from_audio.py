#!/usr/bin/env python
# coding: utf-8

"""
End-to-end helper to run Yandex SpeechKit STT on a long OGG/Opus call recording.

Usage (from project root):
    python -m app.run_speechkit_from_audio [path/to/audio.ogg]

Defaults to: data/video_from_bucket_audio.ogg
Writes plain-text transcript to:
    data/video_from_bucket_audio_speechkit.txt
"""

import os
import sys
import subprocess
from pathlib import Path

from app.speech_processor import (
    SpeechProcessor,
    create_yandex_s3_client_from_env,
    _load_dotenv_if_present,
    ENV_YANDEX_S3_BUCKET,
    ENV_YANDEX_FOLDER_ID,
    ENV_YANDEX_API_KEY,
)


def _run_ffmpeg(args):
    cmd = ["ffmpeg", "-y"] + args
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with code {completed.returncode}: {completed.stderr.decode(errors='ignore')}"
        )


def main():
    project_root = Path(__file__).resolve().parent.parent
    _load_dotenv_if_present(project_root)

    # Input audio (original OGG/Opus from video)
    if len(sys.argv) > 1:
        input_audio = Path(sys.argv[1]).resolve()
    else:
        input_audio = (project_root / "data" / "video_from_bucket_audio.ogg").resolve()

    if not input_audio.is_file():
        print(f"Error: audio file not found: {input_audio}")
        sys.exit(1)

    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    full_mono = data_dir / "video_full_mono.ogg"
    chunks_dir = data_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = data_dir / "video_from_bucket_audio_speechkit.txt"

    # 1) Normalize to mono Opus (helps recognition)
    _run_ffmpeg(
        [
            "-i",
            str(input_audio),
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "libopus",
            str(full_mono),
        ]
    )

    # 2) Split into 30-second chunks
    # Clear old chunks
    for old in chunks_dir.glob("chunk_*.ogg"):
        try:
            old.unlink()
        except OSError:
            pass

    _run_ffmpeg(
        [
            "-i",
            str(full_mono),
            "-f",
            "segment",
            "-segment_time",
            "30",
            "-c",
            "copy",
            str(chunks_dir / "chunk_%03d.ogg"),
        ]
    )

    # 3) Build single SpeechProcessor instance
    s3_client = create_yandex_s3_client_from_env()
    bucket = os.environ[ENV_YANDEX_S3_BUCKET]
    folder_id = os.environ[ENV_YANDEX_FOLDER_ID]
    api_key = os.environ[ENV_YANDEX_API_KEY]

    processor = SpeechProcessor(
        folder_id=folder_id,
        api_key=api_key,
        s3_client=s3_client,
        bucket=bucket,
    )

    # 4) Run STT on each chunk and append plain Russian text
    if transcript_path.is_file():
        transcript_path.unlink()

    chunk_files = sorted(chunks_dir.glob("chunk_*.ogg"))
    if not chunk_files:
        print(f"No chunks produced in {chunks_dir}")
        sys.exit(1)

    with transcript_path.open("a", encoding="utf-8") as out_f:
        for chunk in chunk_files:
            print(f"Processing {chunk.name}...")
            with chunk.open("rb") as f:
                audio_bytes = f.read()
            result = processor.speech_to_text(
                audio_bytes,
                file_extension=".ogg",
                language_code="ru-RU",
            )
            text = ""
            if isinstance(result, dict):
                text = result.get("text") or ""
            if text:
                out_f.write(text + "\n")

    print(f"Transcript saved to: {transcript_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8
"""
Simple helper to test Yandex SpeechKit STT using `SpeechProcessor`.

Usage (from project root):
    python -m app.run_speechkit_from_audio [path/to/audio.ogg]

If no path is provided, uses: data/video_from_bucket_audio.ogg
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

DEFAULT_AUDIO = PROJECT_ROOT / "data" / "video_from_bucket_audio.ogg"


class _DummyS3Client:
    """
    Minimal stub S3 client so `SpeechProcessor` can run without real Object Storage.
    """

    def upload_fileobj(self, fileobj, bucket, key):
        # No-op: just consume the file-like object
        fileobj.read()

    def generate_presigned_url(self, *_args, **_kwargs):
        return None


def _load_env():
    """Best-effort .env loading (matches other helpers in the project)."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        # Fallback: manual parsing if python-dotenv is not installed
        env_path = PROJECT_ROOT / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def main():
    from app.speech_processor import SpeechProcessor

    _load_env()

    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        print(f"Error: audio file not found: {audio_path}")
        return 1

    folder_id = os.environ.get("YANDEX_FOLDER_ID", "")
    api_key = os.environ.get("YANDEX_API_KEY", "")
    if not api_key:
        print("Error: YANDEX_API_KEY is not set (see .env).")
        return 1

    s3_client = _DummyS3Client()
    bucket = os.environ.get("YANDEX_S3_BUCKET", "dummy-bucket")

    processor = SpeechProcessor(
        folder_id=folder_id,
        api_key=api_key,
        s3_client=s3_client,
        bucket=bucket,
    )

    print(f"Running SpeechKit STT on: {audio_path}")
    file_bytes = audio_path.read_bytes()
    result = processor.speech_to_text(file_bytes=file_bytes, file_extension=".ogg")

    print("\n=== STT RESULT ===")
    print(f"status: {result.get('status')}")
    text = result.get("text") or ""
    if text:
        print("text:")
        print(text)
    else:
        print("No text returned.")

    if result.get("status") != "success":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

