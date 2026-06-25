"""
Text extraction module: Yandex SpeechKit STT, S3 helpers, transcription.
"""
from app.backend.text_extraction.text_extraction import (
    transcribe, main,
    SpeechProcessor, create_yandex_s3_client_from_env,
    list_bucket_objects, extract_audio_to_ogg, get_video_path_from_bucket,
    ENV_YANDEX_S3_BUCKET, ENV_YANDEX_S3_ACCESS_KEY_ID,
    ENV_YANDEX_S3_SECRET_ACCESS_KEY, ENV_YANDEX_FOLDER_ID, ENV_YANDEX_API_KEY,
)

__all__ = [
    "transcribe", "main",
    "SpeechProcessor", "create_yandex_s3_client_from_env",
    "list_bucket_objects", "extract_audio_to_ogg", "get_video_path_from_bucket",
    "ENV_YANDEX_S3_BUCKET", "ENV_YANDEX_S3_ACCESS_KEY_ID",
    "ENV_YANDEX_S3_SECRET_ACCESS_KEY", "ENV_YANDEX_FOLDER_ID", "ENV_YANDEX_API_KEY",
]
