# Примеры использования совместного анализа видео+аудио

## 1. Запуск через основной пайплайн

Самый простой способ - использовать основной скрипт пайплайна:

```bash
cd /path/to/cdek

# Запустить полный пайплайн (включая новый этап)
python -m app.run_full_pipeline data/my_video.webm

# Или использовать локальное видео из data/
python -m app.run_full_pipeline
```

Новый этап будет запущен автоматически после анализа аудио и выведет результаты в:
```
data/v3_dense/combined_emotion_analysis.json
```

---

## 2. Запуск анализа отдельно

Если у вас уже есть готовые файлы анализа, можно запустить совместный анализ отдельно:

```bash
# С автоматическим поиском файлов
python -m app.run_combined_emotion_analysis

# С кастомными путями
python -m app.run_combined_emotion_analysis \
  data/v3_dense/analysis_step6_emotion_report.json \
  data/video_from_bucket_audio_diarized_sentiment.json \
  data/my_combined_emotion_output.json \
  0.6 0.4
```

---

## 3. Программное использование в Python

```python
from app.backend.combined_emotion_analysis import run_combined_emotion_analysis

# Базовый пример
result = run_combined_emotion_analysis(
    emotion_report_path="data/v3_dense/analysis_step6_emotion_report.json",
    audio_sentiment_path="data/video_from_bucket_audio_diarized_sentiment.json",
    output_path="data/v3_dense/combined_emotion_analysis.json",
)

# Проверка результатов
print(f"Dominant emotion: {result['summary']['dominant_combined_emotion']}")
print(f"Average valence: {result['summary']['average_combined_valence']}")
print(f"Total intervals: {result['summary']['total_intervals']}")
```

---

## 4. Экспериментирование с весами

Вы можете изменять веса видео и аудио для разных сценариев анализа:

### Сценарий 1: Фокус на мимике (интервью, дебаты)
```bash
python -m app.run_combined_emotion_analysis \
  data/v3_dense/analysis_step6_emotion_report.json \
  data/video_from_bucket_audio_diarized_sentiment.json \
  data/output_video_focused.json \
  0.7 0.3
```

### Сценарий 2: Сбалансированный анализ (обычный случай, по умолчанию)
```bash
python -m app.run_combined_emotion_analysis \
  data/v3_dense/analysis_step6_emotion_report.json \
  data/video_from_bucket_audio_diarized_sentiment.json \
  data/output_balanced.json \
  0.6 0.4
```

### Сценарий 3: Фокус на содержании речи (презентация, лекция)
```bash
python -m app.run_combined_emotion_analysis \
  data/v3_dense/analysis_step6_emotion_report.json \
  data/video_from_bucket_audio_diarized_sentiment.json \
  data/output_audio_focused.json \
  0.4 0.6
```

---

## 5. Анализ результатов в Python

```python
import json
from pathlib import Path

# Загрузить результаты
with open("data/v3_dense/combined_emotion_analysis.json") as f:
    combined_data = json.load(f)

# Получить общую статистику
summary = combined_data['summary']
print(f"Анализирован {summary['total_intervals']} интервалов")
print(f"Общее эмоциональное состояние: {summary['dominant_combined_emotion']}")
print(f"Средняя валентность: {summary['average_combined_valence']}")

# Распределение эмоций
print("\nРаспределение эмоций:")
for emotion, count in summary['emotion_distribution'].items():
    print(f"  {emotion}: {count}")

# Анализ отдельных интервалов
print("\nДетальный анализ первого интервала:")
interval = combined_data['combined_intervals'][0]
print(f"  Интервал: {interval['start_ts']}")

for person_id, emotions in interval['person_emotions'].items():
    print(f"\n  Персона: {person_id}")
    print(f"    Видео эмоция: {emotions['video_emotion']}")
    print(f"    Аудио сентимент: {emotions['audio_sentiment']}")
    print(f"    Итоговая эмоция: {emotions['combined_emotion']}")
    print(f"    Валентность: {emotions['combined_valence']}")
    print(f"    Уверенность: {emotions['combined_confidence']}")
```

---

## 6. Сравнение видео и аудио анализа

```python
import json
from pathlib import Path

with open("data/v3_dense/combined_emotion_analysis.json") as f:
    combined = json.load(f)

# Найти случаи, где видео и аудио сильно отличаются
mismatches = []
for interval in combined['combined_intervals']:
    for person_id, emotions in interval['person_emotions'].items():
        video_emo = emotions['video_emotion']
        audio_sent = emotions['audio_sentiment']
        
        # Если видео показывает "negative", а аудио "positive", это несовпадение
        if (video_emo in ['sadness', 'anger', 'fear'] and 
            audio_sent == 'positive'):
            mismatches.append({
                'interval': interval['start_ts'],
                'person': person_id,
                'video': video_emo,
                'audio': audio_sent,
                'confidence': emotions['combined_confidence']
            })

print(f"Найдено {len(mismatches)} несовпадений видео и аудио:")
for m in mismatches[:5]:
    print(f"  {m['interval']} - {m['person']}: "
          f"видео={m['video']}, аудио={m['audio']} "
          f"(уверенность: {m['confidence']})")
```

---

## 7. Экспорт результатов для визуализации

```python
import json
import csv

with open("data/v3_dense/combined_emotion_analysis.json") as f:
    combined = json.load(f)

# Экспортировать в CSV для анализа в Excel
with open("combined_emotion_analysis.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "start_ts", "person", "video_emotion", "audio_sentiment", 
        "combined_emotion", "combined_valence", "confidence"
    ])
    
    for interval in combined['combined_intervals']:
        ts = interval['start_ts']
        for person_id, emotions in interval['person_emotions'].items():
            writer.writerow([
                ts,
                person_id,
                emotions['video_emotion'],
                emotions['audio_sentiment'],
                emotions['combined_emotion'],
                emotions['combined_valence'],
                emotions['combined_confidence']
            ])

print("Результаты экспортированы в combined_emotion_analysis.csv")
```

---

## 8. Отладка и диагностика

Если что-то не работает, можно включить подробный лог:

```python
from app.backend.combined_emotion_analysis import (
    run_combined_emotion_analysis,
    load_json,
)

# Проверить, загружаются ли файлы
try:
    emotion_data = load_json("data/v3_dense/analysis_step6_emotion_report.json")
    print(f"✓ Emotion report loaded: {len(emotion_data.get('interval_emotion_report', []))} intervals")
except Exception as e:
    print(f"✗ Error loading emotion report: {e}")

try:
    audio_data = load_json("data/video_from_bucket_audio_diarized_sentiment.json")
    print(f"✓ Audio sentiment loaded: {len(audio_data.get('segments', []))} segments")
except Exception as e:
    print(f"✗ Error loading audio sentiment: {e}")

# Запустить анализ с перехватом ошибок
try:
    result = run_combined_emotion_analysis(
        emotion_report_path="data/v3_dense/analysis_step6_emotion_report.json",
        audio_sentiment_path="data/video_from_bucket_audio_diarized_sentiment.json",
        output_path="data/v3_dense/combined_emotion_analysis.json",
    )
    print(f"✓ Analysis completed successfully")
except Exception as e:
    print(f"✗ Analysis failed: {e}")
    import traceback
    traceback.print_exc()
```

---

## FAQ

**Q: Что означает "combined_emotion"?**
A: Это итоговая эмоция, получившаяся из комбинации видео (мимика) и аудио (сентимент речи):
- `very_positive` - очень позитивно (валентность > 0.8)
- `positive` - позитивно (0.4-0.8)
- `neutral` - нейтрально (-0.2-0.4)
- `negative` - негативно (-0.6 до -0.2)
- `very_negative` - очень негативно (< -0.6)

**Q: Почему видео и аудио дают разные результаты?**
A: Это нормально. Человек может улыбаться (видео) но говорить мрачно (аудио), или наоборот. Совместный анализ показывает полную картину.

**Q: Как выбрать правильные веса?**
A: Зависит от контекста:
- Интервью/переговоры: больше вес на видео (0.6-0.7)
- Естественный разговор: равный вес (0.5-0.5)
- Презентация/лекция: больше вес на аудио (0.3-0.4)

**Q: Отличается ли "combined_emotion" от "dominant_emotion"?**
A: Да! `dominant_emotion` - только из видео (мимика), `combined_emotion` - комбинация видео и аудио.

---

Дата создания: 2026-03-05
