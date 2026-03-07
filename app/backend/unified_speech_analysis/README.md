# Unified Speech Analysis Module

Combined module for speech processing using:
- **SpeechKit** (Yandex) for speech-to-text
- **Torch + pyannote** for speaker diarization  
- **Hugging Face transformers** for sentiment analysis

## Architecture

```
UnifiedSpeechAnalyzer
├── transcribe_with_speechkit()     ← Yandex SpeechKit STT
├── get_diarization_torch()         ← Pyannote (Torch-based)
├── get_sentiment_torch()           ← HuggingFace transformers
└── analyze_segments()              ← Combined analysis
```

## Features

1. **Speech Recognition**: Uses Yandex SpeechKit for accurate Russian speech-to-text
2. **Speaker Diarization**: Uses pyannote with Torch backend for speaker identification
3. **Sentiment Analysis**: Uses Hugging Face transformers (RuBERT) for Russian sentiment
4. **GPU Support**: Automatically uses CUDA if available
5. **Fallback Handling**: Alternating speaker assignment if diarization fails

## Usage

### Basic Usage

```python
from app.backend.unified_speech_analysis import UnifiedSpeechAnalyzer
from pathlib import Path

analyzer = UnifiedSpeechAnalyzer()

# Load segments from Whisper
segments = [
    {"start": 0.0, "end": 2.5, "text": "Привет, это тестовый текст"},
    {"start": 2.5, "end": 5.0, "text": "Как дела?"}
]

# Analyze
result = analyzer.analyze_segments(
    Path("audio.ogg"),
    segments,
    role_1="client",
    role_2="company_representative"
)
```

### CLI Usage

```bash
# Run analysis
python -m app.backend.unified_speech_analysis segments.json audio.ogg

# With environment variables
export YANDEX_API_KEY="your-key"
export YANDEX_FOLDER_ID="your-folder"
export YANDEX_S3_BUCKET="your-bucket"
export HUGGINGFACE_TOKEN="your-token"
python -m app.backend.unified_speech_analysis segments.json audio.ogg
```

## Dependencies

Required packages:
```
openai-whisper
pyannote.audio
torch
transformers
```

Install with:
```bash
pip install openai-whisper pyannote.audio torch transformers
```

## Configuration

### Environment Variables

```bash
# Yandex SpeechKit
YANDEX_API_KEY=your-speechkit-api-key
YANDEX_FOLDER_ID=your-folder-id
YANDEX_S3_BUCKET=your-bucket

# HuggingFace (for pyannote)
HUGGINGFACE_TOKEN=your-huggingface-token

# Speaker roles
SPEAKER_1_ROLE=client
SPEAKER_2_ROLE=company_representative

# Fallback behavior
USE_ALTERNATING_SPEAKERS_FALLBACK=1  # Use alternating if diarization fails
```

## Output Format

### JSON Output

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Привет, это тестовый текст",
      "speaker_id": "SPEAKER_1",
      "role": "client",
      "sentiment": {
        "label": "neutral",
        "score": 0.95
      }
    }
  ],
  "speaker_roles": {"SPEAKER_1": "client", "SPEAKER_2": "company_representative"},
  "diarization_fallback_used": false,
  "num_segments": 1,
  "num_speakers": 2
}
```

## GPU Support

The module automatically detects and uses CUDA if available:

```python
# Check available GPU
import torch
print(torch.cuda.is_available())  # True if GPU is available
```

For faster processing, ensure PyTorch is installed with CUDA support:

```bash
# For CUDA 11.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Integration with Full Pipeline

Update `run_full_pipeline.py` to use unified analysis:

```python
from app.backend.unified_speech_analysis import run_unified_analysis

result = run_unified_analysis(
    ogg_path,
    segments,
    role_1=os.environ.get("SPEAKER_1_ROLE", "client"),
    role_2=os.environ.get("SPEAKER_2_ROLE", "company_representative"),
)
```

## Troubleshooting

### Diarization not working
- Ensure `HUGGINGFACE_TOKEN` is set
- Check pyannote is installed: `pip install pyannote.audio`
- Check token has access to pyannote models

### Sentiment analysis not working
- Ensure transformers is installed: `pip install transformers`
- Check internet connection for model download
- Models will be cached after first download

### SpeechKit errors
- Verify Yandex credentials in `.env`
- Check internet connection to Yandex cloud
- Review logs for specific gRPC errors
