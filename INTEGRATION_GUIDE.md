# SpeechKit + Whisper + HuggingFace Integration Guide

## Overview

This document describes the unified speech analysis architecture combining three major components:

1. **Yandex SpeechKit** - Speech-to-Text (STT) Recognition
2. **Whisper (via Torch)** - Text Extraction & Speaker Diarization via pyannote
3. **HuggingFace** - Sentiment Analysis using transformers

## Architecture Diagram

```
Audio Input (OGG/MP3)
    ↓
    ├─→ [SpeechKit] → Recognition (Russian speech → text)
    │                  Used by: speech_processor.py
    │
    ├─→ [Whisper] → Text Extraction & Segmentation
    │               Used by: text_extraction.py
    │
    ├─→ [Torch/Pyannote] → Speaker Diarization
    │                      Used by: unified_speech_analysis.py
    │
    └─→ [HuggingFace/RuBERT] → Sentiment Analysis
                               Used by: unified_speech_analysis.py

Final Output
    ↓
JSON with segments containing:
  - Text
  - Speaker ID & Role
  - Sentiment (label + score)
  - Timing info
```

## Component Details

### 1. Yandex SpeechKit (STT)

**What it does:** Converts speech audio to text using Yandex Cloud's API

**Location:** `app/speech_processor.py`

**Key Classes:**
- `SpeechProcessor` - Main STT handler
  - Method: `speech_to_text()` - Converts audio bytes to text

**Configuration (`.env`):**
```
YANDEX_API_KEY=your-api-key
YANDEX_FOLDER_ID=your-folder-id
YANDEX_S3_BUCKET=your-bucket-name
YANDEX_S3_ACCESS_KEY_ID=key
YANDEX_S3_SECRET_ACCESS_KEY=secret
```

**Usage:**
```python
from app.speech_processor import SpeechProcessor

processor = SpeechProcessor(
    folder_id="your-folder",
    api_key="your-key",
    s3_client=boto3_client,
    bucket="your-bucket"
)

result = processor.speech_to_text(audio_bytes)
print(result["text"])  # Recognized text
```

### 2. Whisper (Text Extraction)

**What it does:** Extracts text from audio with precise timestamps

**Location:** `app/backend/text_extraction/text_extraction.py`

**Key Functions:**
- `transcribe()` - Extracts text and creates segmented output

**Output:**
- `.txt` - Full transcript
- `_segments.json` - Timestamped segments: `[{"start": 0.0, "end": 2.5, "text": "Hello"}]`
- `.timestamped.txt` - Human-readable with timestamps

**Usage:**
```python
from app.backend.text_extraction import transcribe

result = transcribe(
    "audio.ogg",
    model_name="base",      # tiny, small, base, medium, large
    language="ru",
    save_timestamps=True
)

# Returns: {
#   "text": "Full transcript",
#   "segments": [{"start": 0.0, "end": 2.5, "text": "..."}],
#   "output_path": "audio.txt"
# }
```

### 3. Torch + Pyannote (Speaker Diarization)

**What it does:** Identifies which speaker is speaking at each time point

**Location:** `app/backend/unified_speech_analysis/unified_analysis.py`

**Method:** `get_diarization_torch()`

**Key Concept:** Segments incoming audio to speaker identities (SPEAKER_1, SPEAKER_2, etc.)

**Output:** List of `(start, end, speaker_id)` tuples

**Configuration (`.env`):**
```
HUGGINGFACE_TOKEN=your-token
```

**Note:** Requires accepting pyannote license at https://huggingface.co/pyannote/speaker-diarization-community-1

**Usage:**
```python
analyzer = UnifiedSpeechAnalyzer()
diarization = analyzer.get_diarization_torch(
    Path("audio.ogg"),
    num_speakers=2
)
# Returns: [(0.0, 2.5, "SPEAKER_1"), (2.5, 5.0, "SPEAKER_2")]
```

### 4. HuggingFace RuBERT (Sentiment Analysis)

**What it does:** Analyzes sentiment of Russian text

**Model:** `blanchefort/rubert-base-cased-sentiment-rusentiment`

**Location:** `app/backend/unified_speech_analysis/unified_analysis.py`

**Method:** `get_sentiment_torch()`

**Output:** `{"label": "positive/negative/neutral", "score": 0.0-1.0}`

**Features:**
- Automatic GPU detection (uses CUDA if available)
- 512 character truncation to fit model limits
- Russian-optimized (RuBERT)

**Usage:**
```python
analyzer = UnifiedSpeechAnalyzer()
sentiment = analyzer.get_sentiment_torch("Это отличный продукт!")
# Returns: {"label": "positive", "score": 0.95}
```

## Integration Points

### Full Pipeline Workflow

```
1. run_full_pipeline.py
   ├─ Frame extraction & face detection
   ├─ Emotion recognition from video
   │
   ├─ Audio extraction (ffmpeg)
   │
   ├─ Whisper text extraction
   │  └─ Produces: segments.json
   │
   └─ Unified Analysis
      ├─ Input: segments.json + audio.ogg
      ├─ SpeechKit STT [optional]
      ├─ Torch diarization
      ├─ HuggingFace sentiment
      └─ Output: unified_analysis.json
```

### Segment Enrichment

Each Whisper segment is enriched with:

```json
{
  "start": 0.0,
  "end": 2.5,
  "text": "original Whisper text",
  "speaker_id": "SPEAKER_1",           // from diarization
  "role": "client",                    // mapped from SPEAKER_1
  "sentiment": {                       // from sentiment analysis
    "label": "positive",
    "score": 0.87
  }
}
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `app/speech_processor.py` | SpeechKit STT wrapper |
| `app/backend/text_extraction/text_extraction.py` | Whisper transcription |
| `app/backend/unified_speech_analysis/unified_analysis.py` | Main integration module (diarization + sentiment) |
| `app/backend/unified_speech_analysis/example.py` | Example usage script |
| `app/run_full_pipeline.py` | Complete pipeline from video to analysis |

## GPU Acceleration

All components support GPU acceleration:

### SpeechKit
- Uses gRPC streaming (minimal CPU usage)
- No GPU benefits

### Whisper
- Automatically detects CUDA
- ~20-30x faster with GPU
- Uses fp16 precision on A100/V100

### Pyannote (Diarization)
- Uses CUDA automatically
- ~10-15x faster with GPU
- Requires GPU for large files

### RuBERT (Sentiment)
- Automatically uses CUDA
- ~5-10x faster with GPU
- Processes efficiently in batches

**Check GPU availability:**
```python
import torch
print(torch.cuda.is_available())    # True if CUDA available
print(torch.cuda.get_device_name(0)) # GPU model name
```

**Install PyTorch with CUDA:**
```bash
# For CUDA 11.x
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.x  
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Environment Setup

### Complete `.env` Configuration

```bash
# Yandex SpeechKit
YANDEX_API_KEY=your-speechkit-api-key
YANDEX_FOLDER_ID=your-folder-id
YANDEX_S3_BUCKET=your-bucket-name
YANDEX_S3_ACCESS_KEY_ID=your-access-key
YANDEX_S3_SECRET_ACCESS_KEY=your-secret-key

# HuggingFace (for pyannote diarization)
HUGGINGFACE_TOKEN=your-hf-token

# Speaker role mapping
SPEAKER_1_ROLE=client
SPEAKER_2_ROLE=company_representative

# Fallback behavior
USE_ALTERNATING_SPEAKERS_FALLBACK=1  # Use alternating if diarization fails
```

## Dependencies

### Core Requirements
```bash
pip install openai-whisper pyannote.audio torch transformers boto3
```

### Optional
```bash
pip install python-dotenv  # For .env file loading
pip install datasets       # For some HuggingFace operations
```

### Specific Versions (Tested)
```
openai-whisper>=20221117
pyannote.audio>=3.0
torch>=2.0
transformers>=4.30
```

## Troubleshooting

### Diarization fails with "Token is invalid"
- Verify HUGGINGFACE_TOKEN in .env
- Token must have access to pyannote models
- Get token from https://huggingface.co/settings/tokens

### Sentiment analysis downloads large models
- First run downloads ~400MB for RuBERT
- Models cached in ~/.cache/huggingface/
- Subsequent runs are fast

### SpeechKit errors with gRPC
- Verify YANDEX_API_KEY and YANDEX_FOLDER_ID
- Check internet connection to Yandex cloud
- Review gRPC logs for specific errors

### Out of memory on GPU
- Reduce Whisper model size (use `tiny` or `small`)
- Reduce batch processing size
- Use CPU fallback: `CUDA_VISIBLE_DEVICES="" python run_full_pipeline.py`

## Example Outputs

### Unified Analysis JSON
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "text": "Здравствуйте, спасибо за звонок",
      "speaker_id": "SPEAKER_1",
      "role": "client",
      "sentiment": {
        "label": "neutral",
        "score": 0.89
      }
    },
    {
      "start": 3.2,
      "end": 7.5,
      "text": "Добро пожаловать, чем я могу вам помочь?",
      "speaker_id": "SPEAKER_2",
      "role": "company_representative",
      "sentiment": {
        "label": "positive",
        "score": 0.92
      }
    }
  ],
  "speaker_roles": {
    "SPEAKER_1": "client",
    "SPEAKER_2": "company_representative"
  },
  "diarization_fallback_used": false,
  "num_segments": 42,
  "num_speakers": 2
}
```

### Text Output
```
[0.0s - 3.2s] client: Здравствуйте, спасибо за звонок
  sentiment: neutral (0.89)

[3.2s - 7.5s] company_representative: Добро пожаловать, чем я могу вам помочь?
  sentiment: positive (0.92)
```

## Performance Benchmarks

(On Tesla T4 GPU, 60-second audio)

| Component | Time | Notes |
|-----------|------|-------|
| Whisper (base) | 2-3s | With GPU |
| Diarization | 5-8s | 2 speakers |
| Sentiment (42 segments) | 1-2s | Batched |
| **Total** | **8-13s** | Including overhead |

For CPU-only: multiply by 5-10x

## Future Improvements

- [ ] Streaming transcription (for real-time analysis)
- [ ] Multi-speaker sentiment (current: per-segment)
- [ ] Emotion detection from speech tone (prosody analysis)
- [ ] Conversation turn-taking metrics
- [ ] Caching diarization results
- [ ] Model quantization for mobile deployment
- [ ] Real-time WebSocket API wrapper
