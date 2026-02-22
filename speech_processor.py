#!/usr/bin/env python
# coding: utf-8

import io
import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Optional boto3 for Yandex Object Storage (S3-compatible)
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

# Yandex Object Storage (S3-compatible) endpoint
YANDEX_S3_ENDPOINT = "https://storage.yandexcloud.net"
YANDEX_S3_REGION = "ru-central1"

# Env var names for Yandex S3 (set these in .env or shell; never commit secrets)
ENV_YANDEX_BUCKET = "YANDEX_S3_BUCKET"
ENV_YANDEX_ACCESS_KEY = "YANDEX_S3_ACCESS_KEY_ID"
ENV_YANDEX_SECRET_KEY = "YANDEX_S3_SECRET_ACCESS_KEY"


def create_yandex_s3_client(
    bucket_name: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    endpoint_url: str = YANDEX_S3_ENDPOINT,
    region_name: str = YANDEX_S3_REGION,
):
    """
    Create a boto3 S3 client for Yandex Object Storage.

    Use your bucket name and the static access key credentials from
    Yandex Cloud (Console → Service account → Create access key).
    The keys are compatible with AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.

    Args:
        bucket_name: Your Yandex Object Storage bucket name.
        aws_access_key_id: Access Key ID from Yandex Cloud.
        aws_secret_access_key: Secret Access Key from Yandex Cloud.
        endpoint_url: S3 endpoint (default: Yandex).
        region_name: Region (default: ru-central1).

    Returns:
        boto3 S3 client configured for Yandex Object Storage.

    Example:
        s3 = create_yandex_s3_client(
            bucket_name="my-bucket",
            aws_access_key_id="YCAJE...",
            aws_secret_access_key="YCO...",
        )
        s3.upload_file("video.mp4", "my-bucket", "video_files/video.mp4")
    """
    if boto3 is None:
        raise ImportError("boto3 is required for Yandex S3. Install with: pip install boto3")
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def create_yandex_s3_client_from_env(
    bucket_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    endpoint_url: str = YANDEX_S3_ENDPOINT,
    region_name: str = YANDEX_S3_REGION,
):
    """
    Create a Yandex S3 client using environment variables. Keeps secrets out of code.

    Reads (if not passed):
        YANDEX_S3_BUCKET, YANDEX_S3_ACCESS_KEY_ID, YANDEX_S3_SECRET_ACCESS_KEY

    Optional: pip install python-dotenv and create a .env file (see .env.example).
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    bucket = bucket_name or os.environ.get(ENV_YANDEX_BUCKET)
    key_id = aws_access_key_id or os.environ.get(ENV_YANDEX_ACCESS_KEY)
    secret = aws_secret_access_key or os.environ.get(ENV_YANDEX_SECRET_KEY)
    if not bucket or not key_id or not secret:
        raise ValueError(
            "Yandex S3 credentials missing. Set env vars: "
            f"{ENV_YANDEX_BUCKET}, {ENV_YANDEX_ACCESS_KEY}, {ENV_YANDEX_SECRET_KEY}"
        )
    return create_yandex_s3_client(
        bucket_name=bucket,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        endpoint_url=endpoint_url,
        region_name=region_name,
    )


def list_bucket_objects(
    s3_client,
    bucket: str,
    prefix: Optional[str] = None,
    max_keys: Optional[int] = None,
) -> Dict[str, Any]:
    """
    List objects in an S3 bucket (e.g. Yandex Object Storage).

    Args:
        s3_client: Boto3 S3 client.
        bucket: Bucket name.
        prefix: Optional prefix to filter keys (e.g. "video_files/" or "voice_files/").
        max_keys: Optional limit (default: list all).

    Returns:
        Dict with "count" (int) and "keys" (list of str).
    """
    keys: list = []
    paginator = s3_client.get_paginator("list_objects_v2")
    page_kwargs = {"Bucket": bucket}
    if prefix:
        page_kwargs["Prefix"] = prefix
    for page in paginator.paginate(**page_kwargs):
        for obj in page.get("Contents") or []:
            keys.append(obj["Key"])
            if max_keys is not None and len(keys) >= max_keys:
                return {"count": len(keys), "keys": keys}
    return {"count": len(keys), "keys": keys}


def download_file_from_bucket(
    s3_client,
    bucket: str,
    object_key: str,
    local_path: Union[str, Path],
) -> str:
    """
    Download a single file from S3 (Yandex bucket) to a local path.

    Returns:
        Resolved local path as string.
    """
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, object_key, str(path))
    return str(path.resolve())


def get_video_path_from_bucket(
    s3_client=None,
    bucket: Optional[str] = None,
    video_key: Optional[str] = None,
    local_dir: Union[str, Path] = "data",
    prefix: str = "video_files",
) -> str:
    """
    Get a local path to the video: download from Yandex bucket if needed.

    Uses env YANDEX_VIDEO_KEY if set (e.g. video_files/abc.webm), else finds
    the first object under prefix with a video extension (.webm, .mp4, etc.).
    Downloads to local_dir/video_from_bucket.<ext> and returns that path.

    Returns:
        Local file path to the video (existing after download).
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if s3_client is None or bucket is None:
        client = create_yandex_s3_client_from_env()
        bucket = bucket or os.environ.get(ENV_YANDEX_BUCKET)
        if not bucket:
            raise ValueError(f"Set {ENV_YANDEX_BUCKET} or pass bucket")
        s3_client = client
    key = video_key or os.environ.get("YANDEX_VIDEO_KEY")
    if not key:
        listing = list_bucket_objects(s3_client, bucket, prefix=prefix, max_keys=500)
        video_exts = (".webm", ".mp4", ".mov", ".avi", ".mkv")
        for k in listing.get("keys") or []:
            if Path(k).suffix.lower() in video_exts and not k.endswith("/"):
                key = k
                break
        if not key:
            raise FileNotFoundError(
                f"No video file found in bucket under prefix '{prefix}'. "
                "Upload a video or set YANDEX_VIDEO_KEY to the object key."
            )
    ext = Path(key).suffix or ".webm"
    local_path = Path(local_dir) / f"video_from_bucket{ext}"
    download_file_from_bucket(s3_client, bucket, key, local_path)
    return str(local_path.resolve())


def _content_type_for_key(s3_key: str) -> str:
    """Return a Content-Type for common video extensions."""
    ext = Path(s3_key).suffix.lower()
    return {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(ext, "application/octet-stream")


# Yandex SpeechKit SDK imports with proper error handling
try:
    from yandex.cloud.ai.stt.v3 import stt_service_pb2_grpc
    from yandex.cloud.ai.stt.v3 import stt_service_pb2
    import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
    import grpc
except ImportError as e:
    # Mock implementations for environments without Yandex SDK
    stt_service_pb2_grpc = None
    stt_service_pb2 = None
    grpc = None
    print(f"Warning: Yandex SpeechKit SDK not available: {e}")


class SpeechProcessor:
    def __init__(
        self, folder_id: str, api_key: str, s3_client, bucket: str, logger=None
    ):
        """
        Initialize the SpeechProcessor with Yandex SpeechKit credentials and S3 configuration.

        Args:
            folder_id (str): Yandex Cloud folder ID
            api_key (str): Yandex Cloud API key
            s3_client: Boto3 S3 client instance
            bucket (str): S3 bucket name for storing voice files
            logger: Logger instance for logging
        """
        self.folder_id = folder_id
        self.api_key = api_key
        self.s3_client = s3_client
        self.bucket = bucket
        self.logger = logger or logging.getLogger(__name__)

        # Try to initialize Yandex SpeechKit client
        self.stt_client = None
        self.stub = None
        if stt_service_pb2_grpc and grpc:
            try:
                # Create gRPC channel for Yandex SpeechKit
                channel = grpc.secure_channel(
                    "stt.api.cloud.yandex.net:443", grpc.ssl_channel_credentials()
                )
                self.stub = stt_service_pb2_grpc.RecognizerStub(channel)
                self.logger.info("Yandex SpeechKit client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Yandex SpeechKit client: {e}")
        else:
            self.logger.warning(
                "Yandex SpeechKit SDK not available, using mock implementation"
            )

    def upload_voice_file(
        self, file_bytes: bytes, file_extension: str = ".ogg"
    ) -> Dict[str, Any]:
        """
        Upload voice file to S3 storage.

        Args:
            file_bytes (bytes): Voice file content
            file_extension (str): File extension (default: .ogg)

        Returns:
            Dict[str, Any]: Upload result with file metadata
        """
        try:
            # Generate unique file name
            file_id = str(uuid.uuid4())
            s3_key = f"voice_files/{file_id}{file_extension}"

            # Upload to S3
            self.s3_client.upload_fileobj(io.BytesIO(file_bytes), self.bucket, s3_key)

            file_size = len(file_bytes)

            result = {
                "file_id": file_id,
                "s3_bucket": self.bucket,
                "s3_key": s3_key,
                "file_size": file_size,
                "status": "uploaded",
            }

            self.logger.info(f"Voice file uploaded successfully: {file_id}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to upload voice file: {e}")
            return {
                "file_id": None,
                "s3_bucket": self.bucket,
                "s3_key": None,
                "file_size": 0,
                "status": "error",
                "error": str(e),
            }

    def upload_video_file(
        self,
        file_path: Optional[Union[str, Path]] = None,
        file_bytes: Optional[bytes] = None,
        object_key: Optional[str] = None,
        prefix: str = "video_files",
    ) -> Dict[str, Any]:
        """
        Upload a video file to Yandex Object Storage (same bucket as voice files).

        Provide either file_path (for files on disk) or file_bytes (in-memory).
        For large videos, prefer file_path to avoid loading the whole file into memory.

        Args:
            file_path: Path to the video file on disk (e.g. "path/to/video.mp4").
            file_bytes: Video content as bytes (alternative to file_path).
            object_key: Optional S3 object key (e.g. "video_files/my_video.mp4").
                        If not set, a unique key under prefix is generated.
            prefix: S3 key prefix when object_key is not provided (default: video_files).

        Returns:
            Dict with file_id, s3_bucket, s3_key, file_size, status, and optionally error.
        """
        try:
            if object_key:
                s3_key = object_key
                file_id = Path(s3_key).stem
            else:
                file_id = str(uuid.uuid4())
                ext = ""
                if file_path:
                    ext = Path(file_path).suffix
                elif file_bytes and not object_key:
                    ext = ".mp4"  # default for bytes
                s3_key = f"{prefix.rstrip('/')}/{file_id}{ext}"

            if file_path is not None:
                path = Path(file_path)
                if not path.is_file():
                    raise FileNotFoundError(f"Video file not found: {path}")
                self.s3_client.upload_file(
                    str(path),
                    self.bucket,
                    s3_key,
                    ExtraArgs={"ContentType": _content_type_for_key(s3_key)},
                )
                file_size = path.stat().st_size
            elif file_bytes is not None:
                self.s3_client.upload_fileobj(
                    io.BytesIO(file_bytes),
                    self.bucket,
                    s3_key,
                    ExtraArgs={"ContentType": _content_type_for_key(s3_key)},
                )
                file_size = len(file_bytes)
            else:
                raise ValueError("Provide either file_path or file_bytes")

            result = {
                "file_id": file_id,
                "s3_bucket": self.bucket,
                "s3_key": s3_key,
                "file_size": file_size,
                "status": "uploaded",
            }
            self.logger.info(f"Video file uploaded successfully: {s3_key}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to upload video file: {e}")
            return {
                "file_id": None,
                "s3_bucket": self.bucket,
                "s3_key": None,
                "file_size": 0,
                "status": "error",
                "error": str(e),
            }

    def speech_to_text(
        self,
        file_bytes: bytes,
        file_extension: str = ".ogg",
        language_code: str = "ru-RU",
    ) -> Dict[str, Any]:
        """
        Convert speech to text using Yandex SpeechKit streaming API.

        Args:
            file_bytes (bytes): Voice file content
            file_extension (str): File extension (default: .ogg)
            language_code (str): Language code (default: ru-RU)

        Returns:
            Dict[str, Any]: Speech recognition result
        """
        try:
            # First upload to S3 if configured
            upload_result = self.upload_voice_file(file_bytes, file_extension)
            file_id = upload_result["file_id"]

            # Try to recognize with Yandex SpeechKit using streaming API
            if self.stub and stt_pb2:
                CHUNK_SIZE = 4000

                # Define generator function for streaming audio data
                def gen():
                    # Set up recognition options for streaming
                    # For Telegram OGG/OPUS files, use container_audio format
                    recognize_options = stt_pb2.StreamingOptions(
                        recognition_model=stt_pb2.RecognitionModelOptions(
                            audio_format=stt_pb2.AudioFormatOptions(
                                container_audio=stt_pb2.ContainerAudio(
                                    container_audio_type=stt_pb2.ContainerAudio.OGG_OPUS
                                )
                            ),
                            # Set language restriction
                            language_restriction=stt_pb2.LanguageRestrictionOptions(
                                restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                                language_code=[language_code],
                            ),
                            # Use real-time processing
                            audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
                        )
                    )

                    # Send message with recognition settings
                    yield stt_pb2.StreamingRequest(session_options=recognize_options)

                    # Send audio data in chunks
                    for i in range(0, len(file_bytes), CHUNK_SIZE):
                        chunk = file_bytes[i : i + CHUNK_SIZE]
                        yield stt_pb2.StreamingRequest(
                            chunk=stt_pb2.AudioChunk(data=chunk)
                        )

                # Set authentication metadata
                metadata = (("authorization", f"Api-Key {self.api_key}"),)

                # Make the streaming recognition request
                self.logger.info(
                    f"Sending streaming recognition request for file {file_id}"
                )
                it = self.stub.RecognizeStreaming(gen(), metadata=metadata)

                # Process streaming responses
                final_text = ""
                for r in it:
                    event_type = r.WhichOneof("Event")
                    if event_type == "partial" and len(r.partial.alternatives) > 0:
                        # Handle partial results
                        alternatives = [a.text for a in r.partial.alternatives]
                        self.logger.debug(f"Partial result: {alternatives[0]}")
                    elif event_type == "final":
                        # Handle final results
                        alternatives = [a.text for a in r.final.alternatives]
                        if alternatives:
                            final_text = alternatives[0]
                            self.logger.debug(f"Final result: {final_text}")
                    elif event_type == "final_refinement":
                        # Handle final refinement results
                        alternatives = [
                            a.text
                            for a in r.final_refinement.normalized_text.alternatives
                        ]
                        if alternatives:
                            final_text = alternatives[0]
                            self.logger.debug(f"Final refinement result: {final_text}")

                # Return the final recognition result
                if final_text:
                    self.logger.info(
                        f"Speech recognition successful: {len(final_text)} chars"
                    )
                    return {
                        "text": final_text,
                        "status": "success",
                        "file_id": file_id,
                        "s3_key": upload_result.get("s3_key"),
                        "upload_status": upload_result.get("status"),
                        "language": language_code,
                    }
                else:
                    self.logger.warning("Speech recognition returned empty result")
                    return {
                        "text": "",
                        "status": "no_speech",
                        "file_id": file_id,
                        "s3_key": upload_result.get("s3_key"),
                        "upload_status": upload_result.get("status"),
                        "language": language_code,
                    }

        except Exception as e:
            self.logger.error(f"Speech-to-text processing failed: {e}")
            return {"text": "", "status": "error", "error": str(e), "file_id": None}

    def get_voice_file_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate presigned URL for voice file.

        Args:
            s3_key (str): S3 key for the voice file
            expiration (int): URL expiration time in seconds (default: 1 hour)

        Returns:
            Optional[str]: Presigned URL or None if failed
        """
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=expiration,
            )
            return url
        except Exception as e:
            self.logger.error(f"Failed to generate presigned URL: {e}")
            return None
