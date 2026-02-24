#!/usr/bin/env python
# coding: utf-8
"""
Upload a local video file to Yandex Object Storage.
Uses .env for credentials. Default video: project root video2.webm.

Usage (from project root):
    python -m app.upload_video_to_bucket [path/to/video.webm]
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_VIDEO = PROJECT_ROOT / "video2.webm"


def main():
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    video_path = video_path.resolve()
    if not video_path.is_file():
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    from app.speech_processor import (
        create_yandex_s3_client_from_env,
        ENV_YANDEX_BUCKET,
        SpeechProcessor,
    )

    s3_client = create_yandex_s3_client_from_env()
    bucket = os.environ[ENV_YANDEX_BUCKET]
    processor = SpeechProcessor(
        folder_id="",
        api_key="",
        s3_client=s3_client,
        bucket=bucket,
    )

    result = processor.upload_video_file(file_path=str(video_path))
    if result["status"] == "uploaded":
        print(f"Uploaded: {result['s3_key']} ({result['file_size']} bytes)")
    else:
        print(f"Upload failed: {result.get('error', result)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
