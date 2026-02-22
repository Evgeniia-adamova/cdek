#!/usr/bin/env python
# coding: utf-8
"""
Upload a local video file to Yandex Object Storage using existing speech_processor helpers.
Uses .env for credentials (YANDEX_S3_BUCKET, YANDEX_S3_ACCESS_KEY_ID, YANDEX_S3_SECRET_ACCESS_KEY).

Usage:
    python upload_video_to_bucket.py [path/to/video.webm]
    Default: video2.webm in the same folder as this script.
"""

import os
import sys
from pathlib import Path

# Project root = folder containing this script
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = SCRIPT_DIR / "video2.webm"


def main():
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    video_path = video_path.resolve()
    if not video_path.is_file():
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    from speech_processor import (
        create_yandex_s3_client_from_env,
        ENV_YANDEX_BUCKET,
        SpeechProcessor,
    )

    # Connect using env (loads .env if python-dotenv installed)
    s3_client = create_yandex_s3_client_from_env()
    bucket = os.environ[ENV_YANDEX_BUCKET]

    # Use SpeechProcessor's upload_video_file (same bucket/code path as voice)
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
