#!/usr/bin/env python
# coding: utf-8

import io
import os
import sys
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import boto3

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

    def upload_video_file(self, file_path: str) -> Dict[str, Any]:
        """
        Upload video file to S3 storage.

        Args:
            file_path (str): Path to the video file to upload

        Returns:
            Dict[str, Any]: Upload result with file metadata
        """
        try:
            file_path = Path(file_path)
            if not file_path.is_file():
                raise FileNotFoundError(f"Video file not found: {file_path}")
            
            # Generate S3 key
            file_id = str(uuid.uuid4())
            file_ext = file_path.suffix or ".webm"
            s3_key = f"video_files/{file_id}{file_ext}"
            
            # Upload to S3
            file_size = file_path.stat().st_size
            with file_path.open("rb") as f:
                self.s3_client.upload_fileobj(f, self.bucket, s3_key)
            
            result = {
                "file_id": file_id,
                "s3_bucket": self.bucket,
                "s3_key": s3_key,
                "file_size": file_size,
                "status": "uploaded",
            }
            
            self.logger.info(f"Video file uploaded successfully: {file_id}")
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


# Environment variable names (shared helpers for CLI and other modules)
ENV_YANDEX_S3_BUCKET = "YANDEX_S3_BUCKET"
ENV_YANDEX_S3_ACCESS_KEY_ID = "YANDEX_S3_ACCESS_KEY_ID"
ENV_YANDEX_S3_SECRET_ACCESS_KEY = "YANDEX_S3_SECRET_ACCESS_KEY"
ENV_YANDEX_FOLDER_ID = "YANDEX_FOLDER_ID"
ENV_YANDEX_API_KEY = "YANDEX_API_KEY"


def _load_dotenv_if_present(project_root: Path) -> None:
    """
    Load key=value pairs from a .env file in project root into os.environ
    if they are not already set. No external dependencies.
    """
    dotenv_path = project_root / ".env"
    if not dotenv_path.is_file():
        return

    try:
        with dotenv_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Fail silently here; individual getenv calls will error clearly if needed.
        pass


def create_yandex_s3_client_from_env():
    """
    Create a boto3 S3 client for Yandex Object Storage using environment variables.
    """
    endpoint = os.environ.get("YANDEX_S3_ENDPOINT", "https://storage.yandexcloud.net")
    access_key = os.environ[ENV_YANDEX_S3_ACCESS_KEY_ID]
    secret_key = os.environ[ENV_YANDEX_S3_SECRET_ACCESS_KEY]

    session = boto3.session.Session()
    s3_client = session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.environ.get("YANDEX_S3_REGION", "ru-central1"),
    )
    return s3_client


def list_bucket_objects(s3_client, bucket: str, prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    List objects in a Yandex Object Storage bucket, optionally filtered by prefix.
    """
    kwargs: Dict[str, Any] = {"Bucket": bucket}
    if prefix:
        kwargs["Prefix"] = prefix

    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    return {"count": len(keys), "keys": keys}


def _extract_audio_to_ogg(video_path: str, output_path) -> bool:
    """
    Extract audio from video file to OGG format using ffmpeg.

    Args:
        video_path: Path to input video file
        output_path: Path for output .ogg file (str or Path)

    Returns:
        True if extraction succeeded, False otherwise
    """
    import subprocess
    output_path = str(output_path)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "libvorbis",
        "-q:a", "4",
        output_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr.decode(errors='replace')[-500:]}", file=sys.stderr)
            return False
        return True
    except FileNotFoundError:
        print("ffmpeg not found. Install ffmpeg and add it to PATH.", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("ffmpeg timed out.", file=sys.stderr)
        return False


def get_video_path_from_bucket(local_dir: str = "data") -> str:
    """
    Download video from Yandex Object Storage bucket.
    
    Args:
        local_dir (str): Local directory to save the video
        
    Returns:
        str: Path to the downloaded video file
        
    Raises:
        FileNotFoundError: If no video file is found in the bucket
    """
    project_root = Path(__file__).resolve().parent.parent
    _load_dotenv_if_present(project_root)
    
    # Check if required environment variables are set
    if ENV_YANDEX_S3_ACCESS_KEY_ID not in os.environ or ENV_YANDEX_S3_BUCKET not in os.environ:
        raise FileNotFoundError(
            f"Yandex S3 credentials not configured. "
            f"Please set environment variables:\n"
            f"  - {ENV_YANDEX_S3_ACCESS_KEY_ID}\n"
            f"  - {ENV_YANDEX_S3_SECRET_ACCESS_KEY}\n"
            f"  - {ENV_YANDEX_S3_BUCKET}\n"
            f"Or provide a local video file and run: python -m app.run_full_pipeline <video_path>"
        )
    
    s3_client = create_yandex_s3_client_from_env()
    bucket = os.environ[ENV_YANDEX_S3_BUCKET]
    
    # Check for environment variable specifying video key
    video_key = os.environ.get("YANDEX_VIDEO_KEY")
    
    # If no specific video key, search for video files in the bucket
    if not video_key:
        # List potential video folders
        for prefix in ["video_files/", "videos/", ""]:
            result = list_bucket_objects(s3_client, bucket, prefix=prefix)
            
            # Find first video file (looking for common video extensions)
            for key in result["keys"]:
                if key.lower().endswith((".webm", ".mp4", ".avi", ".mov", ".mkv")):
                    video_key = key
                    break
            
            if video_key:
                break
    
    if not video_key:
        raise FileNotFoundError(
            f"No video file found in bucket '{bucket}'. "
            "Set YANDEX_VIDEO_KEY environment variable or ensure video files exist in the bucket."
        )
    
    # Create local directory if it doesn't exist
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Determine file extension from S3 key
    file_ext = Path(video_key).suffix or ".webm"
    local_file = local_path / f"video_from_bucket{file_ext}"
    
    # Download file
    print(f"Downloading {video_key} from bucket...")
    s3_client.download_file(bucket, video_key, str(local_file))
    print(f"Downloaded to: {local_file}")
    
    return str(local_file)


def main():
    """
    CLI entrypoint to run Yandex SpeechKit STT on a local audio file.

    Usage (from project root):
        python -m app.speech_processor [path/to/audio.ogg]

    Defaults to: data/video_from_bucket_audio.ogg
    """
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).resolve().parent.parent
    _load_dotenv_if_present(project_root)

    # Determine audio file path
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1]).resolve()
    else:
        audio_path = (project_root / "data" / "video_from_bucket_audio.ogg").resolve()

    if not audio_path.is_file():
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)

    with audio_path.open("rb") as f:
        audio_bytes = f.read()

    # Build S3 client and SpeechProcessor from environment
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

    result = processor.speech_to_text(
        audio_bytes,
        file_extension=".ogg",
        language_code="ru-RU",
    )

    # If SPEECHKIT_TRANSCRIPT_FILE is set, append plain text there in UTF-8
    transcript_path = os.environ.get("SPEECHKIT_TRANSCRIPT_FILE")
    if transcript_path:
        text = ""
        if isinstance(result, dict):
            text = result.get("text") or ""
        if text:
            with open(transcript_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
    else:
        # Fallback to console output, handling Windows encoding issues
        try:
            print(result)
        except UnicodeEncodeError:
            import json

            sys.stdout.buffer.write(
                (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")
            )


if __name__ == "__main__":
    main()
