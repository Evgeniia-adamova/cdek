# Архитектура совместного анализа видео+аудио

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ВИДЕО ФАЙЛ                                   │
│                    (video.webm, видео.mp4)                          │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ├─────────────────────────────────────────┐
             │                                           │
             ▼                                           ▼
    ┌──────────────────┐                    ┌──────────────────┐
    │  ВИДЕО АНАЛИЗ    │                    │  АУДИО АНАЛИЗ    │
    │  (Step 1-6)      │                    │  (Audio)         │
    └────────┬─────────┘                    └────────┬─────────┘
             │                                       │
             ├─ Step 1: Frame extraction             ├─ Audio extraction
             │  (извлечение кадров)                 │  (ffmpeg)
             │                                       │
             ├─ Step 2: Face detection               ├─ Whisper
             │  (детекция лиц)                      │  (транскрипция)  
             │                                       │
             ├─ Step 3: Face tracking                ├─ Pyannote
             │  (отслеживание лиц)                  │  (дiarization)
             │                                       │
             ├─ Step 4: Dense sampling               ├─ Sentiment analysis
             │  (плотная выборка)                   │  (анализ сентимента)
             │                                       │
             ├─ Step 5: ONNX emotion inference       ▼
             │  (определение эмоций)        ┌───────────────────┐
             │                              │ audio_sentiment    │
             ├─ Step 6: Emotion report      │ _diarized.json     │
             │  (отчет об эмоциях)          │                   │
             │                              │ Сегменты с:       │
             ▼                              │ - start/end       │
    ┌────────────────────┐                 │ - text            │
    │ emotion_report.json│                 │ - sentiment       │
    │                    │                 │ - speaker_id      │
    │ Интервалы с:       │                 └────────┬──────────┘
    │ - frame_count      │                          │
    │ - emotions         │                          │
    │ - valence_score    │◄──────────────────────────┘
    │ - confidence       │
    └────────┬───────────┘
             │
             │     ┌────────────────────────────────────────────────┐
             │     │    COMBINED EMOTION ANALYSIS           │
             │     │                                                 │
             │     │  ┌──────────────────────────────────────────┐  │
             │     │  │ 1. Temporal Alignment                   │  │
             │     │  │    (временное выравнивание)             │  │
             │     │  │    - Соединение видео интервалов        │  │
             │     │  │      с аудио сегментами                 │  │
             │     │  │    - 10-секундные интервалы             │  │
             │     │  └──────────────────┬───────────────────────┘  │
             │     │                     │                           │
             │     │  ┌──────────────────▼───────────────────────┐  │
             │     │  │ 2. Emotion-Sentiment Mapping            │  │
             │     │  │    (маппинг эмоций-сентимента)          │  │
             │     │  │    - Video: neutral→0, happiness→1,     │  │
             │     │  │             sadness→-0.7, anger→-1      │  │
             │     │  │    - Audio: positive→0.7, neutral→0,    │  │
             │     │  │            negative→-0.7                │  │
             │     │  └──────────────────┬───────────────────────┘  │
             │     │                     │                           │
             │     │  ┌──────────────────▼───────────────────────┐  │
             │     │  │ 3. Weighted Combination                 │  │
             │     │  │    (взвешенное объединение)             │  │
             │     │  │    combined_valence =                   │  │
             │     │  │      video_valence * 0.6 +              │  │
             │     │  │      audio_valence * 0.4                │  │
             │     │  │    (веса настраиваются)                 │  │
             │     │  └──────────────────┬───────────────────────┘  │
             │     │                     │                           │
             │     │  ┌──────────────────▼───────────────────────┐  │
             │     │  │ 4. Emotion Category Mapping             │  │
             │     │  │    (обратное отображение)               │  │
             │     │  │    -0.8 to -0.6: very_negative         │  │
             │     │  │    -0.6 to -0.2: negative              │  │
             │     │  │    -0.2 to 0.4:  neutral               │  │
             │     │  │     0.4 to 0.8:  positive              │  │
             │     │  │     0.8 to 1.0:  very_positive         │  │
             │     │  └──────────────────┬───────────────────────┘  │
             │     │                     │                           │
             │     │  ┌──────────────────▼───────────────────────┐  │
             │     │  │ 5. Aggregation & Report Generation      │  │
             │     │  │    (агрегирование и создание отчета)   │  │
             │     │  │    - Per-interval statistics            │  │
             │     │  │    - Per-person analysis                │  │
             │     │  │    - Overall summary                    │  │
             │     │  │    - Emotion distribution               │  │
             │     │  └──────────────────┬───────────────────────┘  │
             │     │                     │                           │
             │     └─────────────────────┼──────────────────────────┘
             │                           │
             ▼                           ▼
    ┌──────────────────────────────────────────────────┐
    │           ИТОГОВЫЙ ОТЧЕТ                         │
    │                                                   │
    │  combined_emotion_analysis.json            │
    │                                                   │
    │  ├─ meta: параметры анализа                     │
    │  │                                               │
    │  ├─ summary: общая статистика                   │
    │  │  ├─ total_intervals: 81                      │
    │  │  ├─ dominant_emotion: "neutral"              │
    │  │  ├─ avg_valence: 0.3859                      │
    │  │  └─ emotion_distribution: {...}              │
    │  │                                               │
    │  └─ combined_intervals: [                       │
    │     {                                            │
    │       "start_ts": "00:00:01.440",               │
    │       "person_emotions": {                      │
    │         "P001": {                               │
    │           "video_emotion": "neutral",           │
    │           "audio_sentiment": "positive",        │
    │           "combined_emotion": "positive",       │
    │           "combined_valence": 0.31,             │
    │           "combined_confidence": 0.89           │
    │         }                                        │
    │       }                                          │
    │     }                                            │
    │   ]                                              │
    │                                                   │
    └──────────────────────────────────────────────────┘
```

## Компоненты системы 🧩

```
app/backend/combined_emotion_analysis/
│
├── combined_emotion_analysis.py
│   ├── load_json()                    - загрузка JSON
│   ├── save_json()                    - сохранение JSON
│   ├── _find_overlapping_segments()   - синхронизация видео/аудио
│   ├── _aggregate_audio_sentiment()   - агрегирование сентимента
│   ├── _compute_combined_emotion_score()  - комбинирование
│   └── run_combined_emotion_analysis()    - основная функция
│
├── __init__.py
│   └── экспорт run_combined_emotion_analysis()
│
└── README.md
    └── полная документация
```

## Потоки данных 🔄

```
1. LOADING
   emotion_report.json ──────┐
                             ├─► load_json()
   audio_sentiment.json ─────┘

2. SYNCHRONIZATION
   Emotion intervals (10-sec) ─┐
                               ├─► _find_overlapping_segments()
   Audio segments (variable) ──┘
                    │
                    ▼
   overlapping_segments = [seg1, seg2, seg3, ...]

3. AGGREGATION
   overlapping_segments ──┐
                          ├─► _aggregate_audio_sentiment()
   (multiple) ────────────┘
                 │
                 ▼
   audio_sentiment_agg = {
     "label": "positive",
     "score": 0.78,
     "by_speaker": {...}
   }

4. COMBINATION
   video_emotion ─┐
                  ├─► _compute_combined_emotion_score()
   audio_sentiment┘
                  └─► combined_emotion_score

5. MAPPING
   combined_valence ─► emotion_category
   (0.31)             (example)
   └─► "positive"

6. AGGREGATION
   all_combined_emotions ─┐
                          ├─► summary statistics
   all_valences ──────────┘
```

## Параметры и конфигурация ⚙️

```
INPUTS:
  emotion_report_path: str          - путь к emotion report (Step 6)
  audio_sentiment_path: str         - путь к audio sentiment JSON
  
PARAMETERS:
  video_weight: float = 0.6         - вес видео (0-1)
  audio_weight: float = 0.4         - вес аудио (0-1)
  
OUTPUTS:
  combined_emotion.json: str        - путь для сохранения результатов
  
INTERNAL CONSTANTS:
  VALENCE_MAPPING = {               - маппинг эмоций на валентность
    "happiness": 1.0,
    "neutral": 0.0,
    "sadness": -0.7,
    ...
  }
```

## Примеры использования архитектуры 📋

### Use Case 1: Интервью (фокус на видео)
```
run_combined_emotion_analysis(
  emotion_report_path="...",
  audio_sentiment_path="...",
  video_weight=0.7,  ← Выше вес видео
  audio_weight=0.3
)
```

### Use Case 2: Лекция (фокус на аудио)
```
run_combined_emotion_analysis(
  emotion_report_path="...",
  audio_sentiment_path="...",
  video_weight=0.3,  ← Ниже вес видео
  audio_weight=0.7
)
```

### Use Case 3: Равномерный анализ
```
run_combined_emotion_analysis(
  emotion_report_path="...",
  audio_sentiment_path="...",
  video_weight=0.5,  ← Равные веса
  audio_weight=0.5
)
```

---

Дата: 2026-03-05
