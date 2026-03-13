# Multimodal AI Analysis of Online Meetings with CDEK Clients

Automated video analysis pipeline for evaluating call-center operator performance.
Processes video recordings through face detection, emotion recognition, speech-to-text, sentiment analysis, and KPI scoring.

## Pipeline Stages

1. **Frame Extraction** — dense sampling of video frames
2. **Face Detection** — locating faces in extracted frames
3. **Emotion Recognition** — classifying facial emotions
4. **Audio Extraction** — separating audio track from video
5. **Speech-to-Text** — transcription via Whisper / SpeechKit
6. **Text Analysis** — diarization and sentiment scoring
7. **Combined Emotion Analysis** — merging video and audio emotion signals
8. **KPI Evaluation** — final operator performance metrics

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in API keys
```

### Download required models

Some ONNX models are too large for Git and must be downloaded manually:

| Model | Size | Location | Download |
|-------|------|----------|----------|
| ArcFace R100 (fp32) | 249 MB | `app/backend/face_detection/models/arcface_r100_fp32.onnx` | [InsightFace](https://github.com/deepinsight/insightface/tree/master/model_zoo) |

The following models are included in the repo:
- `face_detection_yunet_2023mar.onnx` (228 KB) — YuNet face detector
- `emotion-ferplus-8.onnx` (34 MB) — FERPlus emotion classifier

```bash
python -m app.run_full_pipeline path/to/video.webm
```

## Project Structure

```
app/
  run_full_pipeline.py          # main entry point
  backend/
    face_detection/             # face detection module
    emotion_recognition/        # facial emotion classification
    text_extraction/            # Whisper-based transcription
    text_analysis/              # sentiment & diarization
    combined_emotion_analysis/  # audio + video emotion fusion
    unified_speech_analysis/    # SpeechKit, Whisper, HuggingFace
    calls_kpi_evaluation/       # KPI scoring
    speech_processor.py         # audio processing utilities
data/                           # per-video output (gitignored)
logs/                           # per-video logs (gitignored)
```

## Team

| Name | GitHub |
|------|--------|
| Evgeniia Adamova | [@Evgeniia-adamova](https://github.com/Evgeniia-adamova) |
| Alina Mukhametzyanova | [@ranroose](https://github.com/ranroose) |
| Ekaterina Voronina | [@vekaterina732](https://github.com/vekaterina732) |
