#!/usr/bin/env python
# coding: utf-8
"""
List and count files in the Yandex bucket. Uses .env for credentials.

Usage:
    python list_bucket.py [prefix] [--count-only]
    prefix: optional, e.g. video_files/ or voice_files/
    --count-only: print only the count, no key listing
"""

import os
import sys
from speech_processor import (
    create_yandex_s3_client_from_env,
    ENV_YANDEX_BUCKET,
    list_bucket_objects,
)


def main():
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
