"""
Text extraction module: Yandex SpeechKit STT, S3 helpers, audio utilities, transcription.

Combines SpeechProcessor (streaming + async STT), S3 upload/download,
ffmpeg audio extraction, and the high-level transcribe() function.
"""
import io
import json
import os
import subprocess
import sys
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3


# Yandex SpeechKit SDK imports
try:
    from yandex.cloud.ai.stt.v3 import stt_service_pb2_grpc
    from yandex.cloud.ai.stt.v3 import stt_service_pb2
    import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
    import grpc
except ImportError as e:
    stt_service_pb2_grpc = None
    stt_service_pb2 = None
    grpc = None
    print(f"Warning: Yandex SpeechKit SDK not available: {e}")


# ═══════════════════════════════════════════════════════════════
# Environment variable names
# ═══════════════════════════════════════════════════════════════

ENV_YANDEX_S3_BUCKET = "YANDEX_S3_BUCKET"
ENV_YANDEX_S3_ACCESS_KEY_ID = "YANDEX_S3_ACCESS_KEY_ID"
ENV_YANDEX_S3_SECRET_ACCESS_KEY = "YANDEX_S3_SECRET_ACCESS_KEY"
ENV_YANDEX_FOLDER_ID = "YANDEX_FOLDER_ID"
ENV_YANDEX_API_KEY = "YANDEX_API_KEY"


# ═══════════════════════════════════════════════════════════════
# S3 helpers
# ═══════════════════════════════════════════════════════════════

def create_yandex_s3_client_from_env():
    """Create a boto3 S3 client for Yandex Object Storage using environment variables."""
    endpoint = os.environ.get("YANDEX_S3_ENDPOINT", "https://storage.yandexcloud.net")
    access_key = os.environ[ENV_YANDEX_S3_ACCESS_KEY_ID]
    secret_key = os.environ[ENV_YANDEX_S3_SECRET_ACCESS_KEY]

    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.environ.get("YANDEX_S3_REGION", "ru-central1"),
    )


def list_bucket_objects(s3_client, bucket: str, prefix: Optional[str] = None) -> Dict[str, Any]:
    """List objects in a Yandex Object Storage bucket, optionally filtered by prefix."""
    kwargs: Dict[str, Any] = {"Bucket": bucket}
    if prefix:
        kwargs["Prefix"] = prefix

    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    return {"count": len(keys), "keys": keys}


# ═══════════════════════════════════════════════════════════════
# Audio/video utilities
# ═══════════════════════════════════════════════════════════════

def extract_audio_to_ogg(video_path: str, output_path) -> bool:
    """Extract audio from video file to OGG format using ffmpeg."""
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
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
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
    """Download video from Yandex Object Storage bucket."""
    project_root = Path(__file__).resolve().parents[3]
    _load_dotenv_if_present(project_root)

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

    video_key = os.environ.get("YANDEX_VIDEO_KEY")

    if not video_key:
        for prefix in ["video_files/", "videos/", ""]:
            result = list_bucket_objects(s3_client, bucket, prefix=prefix)
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

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    file_ext = Path(video_key).suffix or ".webm"
    local_file = local_path / f"video_from_bucket{file_ext}"

    print(f"Downloading {video_key} from bucket...")
    s3_client.download_file(bucket, video_key, str(local_file))
    print(f"Downloaded to: {local_file}")

    return str(local_file)


def _load_dotenv_if_present(project_root: Path) -> None:
    """Load .env file into os.environ if not already set."""
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
        pass


# ═══════════════════════════════════════════════════════════════
# SpeechProcessor class
# ═══════════════════════════════════════════════════════════════

class SpeechProcessor:
    """Yandex SpeechKit STT client with S3 file management."""

    def __init__(self, folder_id: str, api_key: str, s3_client, bucket: str, logger=None):
        self.folder_id = folder_id
        self.api_key = api_key
        self.s3_client = s3_client
        self.bucket = bucket
        self.logger = logger or logging.getLogger(__name__)

        self.stub = None
        if stt_service_pb2_grpc and grpc:
            try:
                channel = grpc.secure_channel(
                    "stt.api.cloud.yandex.net:443", grpc.ssl_channel_credentials()
                )
                self.stub = stt_service_pb2_grpc.RecognizerStub(channel)
                self.logger.info("Yandex SpeechKit client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Yandex SpeechKit client: {e}")
        else:
            self.logger.warning("Yandex SpeechKit SDK not available, using mock implementation")

    def upload_voice_file(self, file_bytes: bytes, file_extension: str = ".ogg", public: bool = False) -> Dict[str, Any]:
        """Upload voice file to S3 storage."""
        try:
            file_id = str(uuid.uuid4())
            s3_key = f"voice_files/{file_id}{file_extension}"

            extra_args = {"ACL": "public-read"} if public else {}
            self.s3_client.upload_fileobj(
                io.BytesIO(file_bytes), self.bucket, s3_key, ExtraArgs=extra_args
            )

            return {
                "file_id": file_id, "s3_bucket": self.bucket, "s3_key": s3_key,
                "file_size": len(file_bytes), "status": "uploaded",
            }
        except Exception as e:
            self.logger.error(f"Failed to upload voice file: {e}")
            return {
                "file_id": None, "s3_bucket": self.bucket, "s3_key": None,
                "file_size": 0, "status": "error", "error": str(e),
            }

    def upload_video_file(self, file_path: str) -> Dict[str, Any]:
        """Upload video file to S3 storage."""
        try:
            file_path = Path(file_path)
            if not file_path.is_file():
                raise FileNotFoundError(f"Video file not found: {file_path}")

            file_id = str(uuid.uuid4())
            file_ext = file_path.suffix or ".webm"
            s3_key = f"video_files/{file_id}{file_ext}"

            file_size = file_path.stat().st_size
            with file_path.open("rb") as f:
                self.s3_client.upload_fileobj(f, self.bucket, s3_key)

            return {
                "file_id": file_id, "s3_bucket": self.bucket, "s3_key": s3_key,
                "file_size": file_size, "status": "uploaded",
            }
        except Exception as e:
            self.logger.error(f"Failed to upload video file: {e}")
            return {
                "file_id": None, "s3_bucket": self.bucket, "s3_key": None,
                "file_size": 0, "status": "error", "error": str(e),
            }

    def speech_to_text(self, file_bytes: bytes, file_extension: str = ".ogg", language_code: str = "ru-RU") -> Dict[str, Any]:
        """Convert speech to text using Yandex SpeechKit streaming API."""
        try:
            upload_result = self.upload_voice_file(file_bytes, file_extension)
            file_id = upload_result["file_id"]

            if self.stub and stt_pb2:
                CHUNK_SIZE = 4000

                def gen():
                    recognize_options = stt_pb2.StreamingOptions(
                        recognition_model=stt_pb2.RecognitionModelOptions(
                            audio_format=stt_pb2.AudioFormatOptions(
                                container_audio=stt_pb2.ContainerAudio(
                                    container_audio_type=stt_pb2.ContainerAudio.OGG_OPUS
                                )
                            ),
                            language_restriction=stt_pb2.LanguageRestrictionOptions(
                                restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                                language_code=[language_code],
                            ),
                            audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
                        )
                    )
                    yield stt_pb2.StreamingRequest(session_options=recognize_options)
                    for i in range(0, len(file_bytes), CHUNK_SIZE):
                        chunk = file_bytes[i : i + CHUNK_SIZE]
                        yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=chunk))

                metadata = (("authorization", f"Api-Key {self.api_key}"),)
                self.logger.info(f"Sending streaming recognition request for file {file_id}")
                it = self.stub.RecognizeStreaming(gen(), metadata=metadata)

                final_text = ""
                for r in it:
                    event_type = r.WhichOneof("Event")
                    if event_type == "partial" and len(r.partial.alternatives) > 0:
                        pass
                    elif event_type == "final":
                        alternatives = [a.text for a in r.final.alternatives]
                        if alternatives:
                            final_text = alternatives[0]
                    elif event_type == "final_refinement":
                        alternatives = [a.text for a in r.final_refinement.normalized_text.alternatives]
                        if alternatives:
                            final_text = alternatives[0]

                if final_text:
                    self.logger.info(f"Speech recognition successful: {len(final_text)} chars")
                    return {
                        "text": final_text, "status": "success", "file_id": file_id,
                        "s3_key": upload_result.get("s3_key"),
                        "upload_status": upload_result.get("status"), "language": language_code,
                    }
                else:
                    self.logger.warning("Speech recognition returned empty result")
                    return {
                        "text": "", "status": "no_speech", "file_id": file_id,
                        "s3_key": upload_result.get("s3_key"),
                        "upload_status": upload_result.get("status"), "language": language_code,
                    }

        except Exception as e:
            self.logger.error(f"Speech-to-text processing failed: {e}")
            return {"text": "", "status": "error", "error": str(e), "file_id": None}

    def async_speech_to_text(self, s3_key: str, language_code: str = "ru-RU",
                              poll_interval: int = 5, max_wait: int = 600) -> Dict[str, Any]:
        """Transcribe audio using Yandex SpeechKit async (long-running) API."""
        import time
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retries = Retry(total=5, backoff_factor=2, status_forcelist=[502, 503, 504],
                        allowed_methods=["GET", "POST"])
        session.mount("https://", HTTPAdapter(max_retries=retries))

        s3_uri = self.get_voice_file_url(s3_key, expiration=3600)
        if not s3_uri:
            s3_uri = f"https://storage.yandexcloud.net/{self.bucket}/{s3_key}"

        headers = {"Authorization": f"Api-Key {self.api_key}"}
        body = {
            "config": {
                "specification": {
                    "languageCode": language_code, "model": "general",
                    "audioEncoding": "OGG_OPUS", "literature_text": True,
                }
            },
            "audio": {"uri": s3_uri},
        }

        self.logger.info(f"Submitting async STT request for {s3_key}")
        try:
            resp = session.post(
                "https://transcribe.api.cloud.yandex.net/speech/stt/v2/longRunningRecognize",
                headers=headers, json=body, timeout=60,
            )
        except requests.exceptions.RequestException as e:
            return {"text": "", "segments": [], "status": "error", "error": f"Submit request failed: {e}"}
        if resp.status_code != 200:
            return {"text": "", "segments": [], "status": "error",
                    "error": f"Submit failed ({resp.status_code}): {resp.text[:500]}"}

        operation_id = resp.json().get("id")
        if not operation_id:
            return {"text": "", "segments": [], "status": "error",
                    "error": f"No operation ID in response: {resp.text[:500]}"}

        self.logger.info(f"Operation submitted: {operation_id}")

        operation_url = f"https://operation.api.cloud.yandex.net/operations/{operation_id}"
        elapsed = 0
        data = {}
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            try:
                poll_resp = session.get(operation_url, headers=headers, timeout=60)
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Poll request error ({e}), retrying...")
                continue
            if poll_resp.status_code != 200:
                self.logger.warning(f"Poll failed ({poll_resp.status_code}), retrying...")
                continue
            data = poll_resp.json()
            if data.get("done"):
                self.logger.info(f"Operation completed after {elapsed}s")
                break
        else:
            return {"text": "", "segments": [], "status": "timeout",
                    "error": f"Operation not completed after {max_wait}s"}

        if "error" in data:
            return {"text": "", "segments": [], "status": "error", "error": str(data["error"])}

        response = data.get("response", {})
        chunks = response.get("chunks", [])

        segments = []
        all_texts = []
        for chunk in chunks:
            alternatives = chunk.get("alternatives", [])
            if not alternatives:
                continue
            alt = alternatives[0]
            text = alt.get("text", "").strip()
            if not text:
                continue

            words = alt.get("words", [])
            if words:
                seg_start = self._parse_duration(words[0].get("startTime", "0s"))
                seg_end = self._parse_duration(words[-1].get("endTime", "0s"))
            else:
                seg_start = 0.0
                seg_end = 0.0

            segments.append({"start": round(seg_start, 2), "end": round(seg_end, 2), "text": text})
            all_texts.append(text)

        full_text = " ".join(all_texts)
        self.logger.info(f"Async STT complete: {len(full_text)} chars, {len(segments)} segments")

        self._delete_s3_object(s3_key)

        return {"text": full_text, "segments": segments, "status": "success"}

    def _delete_s3_object(self, s3_key: str) -> None:
        """Delete an S3 object (best-effort)."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            self.logger.info(f"Deleted S3 object: {s3_key}")
        except Exception as e:
            self.logger.warning(f"Failed to delete S3 object {s3_key}: {e}")

    @staticmethod
    def _parse_duration(value) -> float:
        """Parse Yandex duration format to float seconds."""
        if isinstance(value, str):
            return float(value.rstrip("s") or "0")
        if isinstance(value, dict):
            seconds = float(value.get("seconds", 0) or 0)
            nanos = float(value.get("nanos", 0) or 0)
            return seconds + nanos / 1_000_000_000
        return float(value or 0)

    def get_voice_file_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """Generate presigned URL for voice file."""
        try:
            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=expiration,
            )
        except Exception as e:
            self.logger.error(f"Failed to generate presigned URL: {e}")
            return None


# ═══════════════════════════════════════════════════════════════
# High-level transcription
# ═══════════════════════════════════════════════════════════════

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
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    folder_id = os.environ.get(ENV_YANDEX_FOLDER_ID, "")
    api_key = os.environ.get(ENV_YANDEX_API_KEY, "")
    if not api_key:
        raise EnvironmentError(f"{ENV_YANDEX_API_KEY} is not set. Configure it in .env")

    s3_client = create_yandex_s3_client_from_env()
    bucket = os.environ.get(ENV_YANDEX_S3_BUCKET, "")

    processor = SpeechProcessor(
        folder_id=folder_id, api_key=api_key, s3_client=s3_client, bucket=bucket,
    )

    # 1) Normalize to mono Opus
    normalized = path.parent / f"{path.stem}_mono.ogg"
    if verbose:
        print("Normalizing audio to mono OGG/Opus...")
    _normalize_to_opus(path, normalized)

    try:
        # 2) Upload to S3
        if verbose:
            print("Uploading audio to S3...")
        audio_bytes = normalized.read_bytes()
        upload_result = processor.upload_voice_file(audio_bytes, file_extension=".ogg", public=True)

        if upload_result.get("status") != "uploaded":
            raise RuntimeError(f"S3 upload failed: {upload_result.get('error', 'unknown')}")

        s3_key = upload_result["s3_key"]
        if verbose:
            print(f"Uploaded to S3: {s3_key}")

        # 3) Run async recognition
        if verbose:
            print("Running async SpeechKit recognition (this may take a few minutes)...")

        result = processor.async_speech_to_text(s3_key=s3_key, language_code=language)
    finally:
        if normalized.is_file() and normalized != path:
            normalized.unlink(missing_ok=True)

    if result.get("status") != "success":
        raise RuntimeError(f"Async STT failed: {result.get('error', result.get('status'))}")

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
            json.dumps({"language": language, "segments": segments}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        out_ts_txt = base.parent / f"{base.name}.timestamped.txt"
        lines = [f"[{_format_ts(s['start'])} - {_format_ts(s['end'])}] {s['text']}" for s in segments]
        out_ts_txt.write_text("\n".join(lines), encoding="utf-8")

    return {
        "text": full_text, "language": language,
        "output_path": str(out_txt), "segments": segments,
    }


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

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
    result = transcribe(audio_path, output_path=audio_path.with_suffix(".txt"), verbose=True)
    print("\n=== Transcription ===")
    print("Output file:", result["output_path"])
    print("Text length:", len(result["text"]), "chars")
    print("Segments:", len(result["segments"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
