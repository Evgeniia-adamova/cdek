#!/usr/bin/env python
# coding: utf-8
"""
List and count files in the Yandex bucket. Uses .env for credentials.

Usage (from project root):
    python -m app.list_bucket [prefix] [--count-only]
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from app.speech_processor import (
        create_yandex_s3_client_from_env,
        ENV_YANDEX_BUCKET,
        list_bucket_objects,
    )

    args = [a for a in sys.argv[1:] if a != "--count-only"]
    count_only = "--count-only" in sys.argv
    prefix = args[0].strip() if args else None

    s3 = create_yandex_s3_client_from_env()
    bucket = os.environ[ENV_YANDEX_BUCKET]

    result = list_bucket_objects(s3, bucket, prefix=prefix)
    print(f"Bucket: {bucket}")
    if prefix:
        print(f"Prefix: {prefix}")
    print(f"Total objects: {result['count']}")
    if not count_only and result["keys"]:
        for k in result["keys"]:
            print(f"  {k}")


if __name__ == "__main__":
    main()
