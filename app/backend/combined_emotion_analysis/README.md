# Combined Video+Audio Emotion Analysis

## Описание

Новый этап в пайплайне объединяет анализ эмоций с видео и анализ сентимента с аудио для создания комплексного отчёта об эмоциональном состоянии.

## Компоненты

### 1. Модуль `combined_emotion_analysis.py`

Основной модуль с функциями для совместного анализа:

- **`run_combined_emotion_analysis()`** - основная функция, которая:
  - Загружает отчёт об эмоциях с видео (Step 6)
  - Загружает результаты анализа сентимента с аудио (дiarизация + sentiment)
  - Синхронизирует временные интервалы
  - Комбинирует эмоции со сраиванием сентимента
  - Создаёт итоговый отчет

Добавлен новый этап в `run_full_pipeline.py`:

```python
# --- Combined video+audio emotion analysis ---
print("\n=== COMBINED VIDEO+AUDIO EMOTION ANALYSIS ===")
from app.backend.combined_emotion_analysis import run_combined_emotion_analysis
combined_result = run_combined_emotion_analysis(
    emotion_report_path=...,
    audio_sentiment_path=...,
    output_path="data/v3_dense/combined_emotion_analysis.json",
    video_weight=0.6,  # Video эмоции - 60% влияния
    audio_weight=0.4,  # Audio сентимент - 40% влияния
)
```

### 3. Независимый скрипт

Файл `run_combined_emotion_analysis.py` позволяет запустить анализ отдельно:

```bash
# Автоматический поиск файлов (по умолчанию)
python -m app.run_combined_emotion_analysis

# С кастомными путями и весами
python -m app.run_combined_emotion_analysis \
  data/v3_dense/analysis_step6_emotion_report.json \
  data/video_from_bucket_audio_diarized_sentiment.json \
  data/v3_dense/combined_emotion_analysis.json \
  0.6 0.4
```

## Структура выходных данных

Выходной файл `combined_emotion_analysis.json` содержит:

```json
{
  "meta": {
    "emotion_report_path": "...",
    "audio_sentiment_path": "...",
    "video_weight": 0.6,
    "audio_weight": 0.4
  },
  "summary": {
    "total_intervals": 42,
    "dominant_combined_emotion": "positive",
    "average_combined_valence": 0.25,
    "emotion_distribution": {
      "positive": 20,
      "neutral": 15,
      "negative": 7
    }
  },
  "combined_intervals": [
    {
      "interval_id": 0,
      "start_ts": "00:00:01.440",
      "person_emotions": {
        "P001": {
          "total_detections": 6,
          "emotion_counts": { "neutral": 5, "happiness": 1 },
          "dominant_emotion": "neutral",
          "combined_emotion": "neutral",
          "combined_valence": 0.15,
          "combined_confidence": 0.65,
          "video_emotion": "neutral",
          "video_valence": 0.0,
          "video_confidence": 0.8406,
          "audio_sentiment": "neutral",
          "audio_valence": 0.0,
          "audio_confidence": 0.52,
          "audio_sentiment_info": { ... },
          "overlapping_segments_count": 3
        }
      },
      "audio_sentiment": {
        "sentiment_label": "neutral",
        "sentiment_score": 0.52,
        "total_segments": 3,
        "by_speaker": { ... }
      }
    }
  ]
}
```

### Ключевые поля

- **`combined_emotion`** - итоговая эмоция (combined):
  - `very_positive` (valence >= 0.8)
  - `positive` (valence >= 0.4)
  - `neutral` (valence >= -0.2)
  - `negative` (valence >= -0.6)
  - `very_negative` (valence < -0.6)

- **`combined_valence`** - общая оценка эмоционального состояния [-1, 1]
- **`combined_confidence`** - уверенность в анализе [0, 1]
- **`video_emotion`** vs **`audio_sentiment`** - исходные значения с сравнением
- **`overlapping_segments_count`** - количество аудио сегментов, перекрывающих видео интервал

## Веса анализа (Video vs Audio)

Для гибкости анализа поддерживаются разные сценарии взвешивания:

| Сценарий | Video | Audio | Применение |
|----------|-------|-------|-----------|
| **Video-focused** | 0.7 | 0.3 | Когда мимика лица более важна (интервью, дебаты) |
| **Balanced** (default) | 0.6 | 0.4 | Равномерный анализ (большинство случаев) |
| **Neutral** | 0.5 | 0.5 | Полностью одинаковый вес |
| **Audio-focused** | 0.4 | 0.6 | Когда содержание речи более важно (презентация) |
| **Audio-dominant** | 0.3 | 0.7 | Когда аудио основной источник (лекция, подкаст) |

## Алгоритм синхронизации

1. **Временное выравнивание**: Для каждого видео интервала (обычно 10 сек) находятся все аудио сегменты, перекрывающиеся по времени
2. **Агрегирование сентимента**: Сегменты внутри одного интервала усредняются по senti мент-оценкам
3. **Валентность**: Каждая эмоция и сентимент преобразуются в шкалу валентности [-1, 1]
4. **Взвешенное объединение**: Итоговая валентность вычисляется как:
   ```
   combined_valence = (video_valence * video_weight + audio_valence * audio_weight) / (total_weight)
   ```
5. **Обратное отображение**: Валентность преобразуется обратно в категорийную эмоцию

## Примеры использования

### От видео/аудио расположения эмоций к комбинированному анализу:

```python
from app.backend.combined_emotion_analysis import run_combined_emotion_analysis

result = run_combined_emotion_analysis(
    emotion_report_path="data/v3_dense/analysis_step6_emotion_report.json",
    audio_sentiment_path="data/video_from_bucket_audio_diarized_sentiment.json",
    output_path="data/v3_dense/combined_emotion_analysis.json",
    video_weight=0.6,
    audio_weight=0.4
)

# Результат содержит:
print(f"Dominant emotion: {result['summary']['dominant_combined_emotion']}")
print(f"Average valence: {result['summary']['average_combined_valence']}")
```

## Интеграция в уже существующие пайплайны

Если у вас есть уже готовые файлы:
- `analysis_step6_emotion_report.json` (эмоции с видео)
- `video_from_bucket_audio_diarized_sentiment.json` (сентимент с аудио)

Можно запустить анализ отдельно:

```bash
cd /path/to/cdek
python -m app.run_combined_emotion_analysis
```

## Расширение функционала

Функции модуля достаточно гибкие для расширения:

### Добавление своих метrик

Отредактируйте функцию `_compute_combined_emotion_score()` для добавления новых критериев:

```python
def _compute_combined_emotion_score(video_emotion, audio_sentiment, ...):
    # Добавьте собственную логику
    combined_score = ...
    return {...}
```

### Изменение маппинга эмоций

Отредактируйте словари `emotion_valence_map` и `sentiment_valence_map` для своих категорий.

---

**Дата добавления**: 2026-03-05
**Статус**: Production-ready
