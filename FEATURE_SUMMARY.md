# Совместный анализ видео+аудио для эмоций - Итоговый отчет

## Что было реализовано ✅

### 1. **Новый модуль: `combined_emotion_analysis`**
   - **Путь**: `app/backend/combined_emotion_analysis/`
   - **Основной файл**: `combined_emotion_analysis.py`
   
   **Функционал:**
   - Загрузка и синхронизация видео и аудио анализов
   - Временное выравнивание интервалов видео с сегментами аудио
   - Комбинирование эмоций (мимика) с сентиментом (речь)
   - Вычисление комбинированной валентности [-1, 1]
   - Генерация итогового отчета с подробной статистикой

### 2. **Интеграция в пайплайн**
   - **Файл**: `app/run_full_pipeline.py`
   - **Автоматический запуск** в конце полного пайплайна
   - Выходной файл: `data/v3_dense/combined_emotion_analysis.json`

### 3. **Независимый скрипт для запуска анализа**
   - **Файл**: `app/run_combined_emotion_analysis.py`
   - Запуск без полного пайплайна (для переанализа существующих данных)
   - Поддержка кастомных путей и весов

### 4. **Документация и примеры**
   - `README.md` в модуле - подробное описание функционала
   - `EXAMPLES_COMBINED_EMOTION.md` - практические примеры использования

---

## Как использовать 🚀

### Вариант 1: Запустить весь пайплайн (рекомендуется)
```bash
cd /Users/ekaterinavoronina/CDEK\ uni/cdek
python -m app.run_full_pipeline data/my_video.webm
```
Совместный анализ запустится автоматически на последнем этапе.

### Вариант 2: Запустить только совместный анализ
```bash
python -m app.run_combined_emotion_analysis
```
Используя существующие файлы:
- Эмоции видео: `data/v3_dense/analysis_step6_emotion_report.json`
- Сентимент аудио: `data/video_from_bucket_audio_diarized_sentiment.json`

### Вариант 3: Использовать в Python коде
```python
from app.backend.combined_emotion_analysis import run_combined_emotion_analysis

result = run_combined_emotion_analysis(
    emotion_report_path="...",
    audio_sentiment_path="...",
    output_path="...",
    video_weight=0.6,  # 60% видео
    audio_weight=0.4   # 40% аудио
)
```

---

## Выходные данные 📊

Файл `combined_emotion_analysis.json` содержит:

```
├── meta (информация о входах и параметрах)
├── summary (общая статистика)
│   ├── total_intervals: 81
│   ├── dominant_combined_emotion: "neutral"
│   ├── average_combined_valence: 0.3859
│   └── emotion_distribution: {...}
└── combined_intervals (подробный анализ каждого интервала)
    └── для каждого интервала:
        ├── interval_id, start_ts
        ├── person_emotions (для каждого человека)
        │   ├── combined_emotion: "positive"
        │   ├── combined_valence: 0.31
        │   ├── combined_confidence: 0.89
        │   ├── video_emotion: "neutral" (из видео)
        │   ├── audio_sentiment: "positive" (из аудио)
        │   └── overlapping_segments_count: 3
        └── audio_sentiment (агрегированный сентимент интервала)
```

### Категории `combined_emotion`:
- **`very_positive`** - валентность > 0.8
- **`positive`** - валентность 0.4-0.8
- **`neutral`** - валентность -0.2 до 0.4
- **`negative`** - валентность -0.6 до -0.2
- **`very_negative`** - валентность < -0.6

---

## Главные особенности 🎯

1. **Синхронизация по времени**: Точное выравнивание 10-секундных видео интервалов с аудио сегментами
2. **Гибкие веса**: Легко переключаться между фокусом на видео или аудио
3. **Подробная статистика**: Для каждого интервала и персоны
4. **Валентность**: Единая шкала -1 до +1 для сравнения
5. **Обработка ошибок**: Корректная обработка пропущенных данных

---

## Примеры выходных результатов 📈

### Пример 1: Интервал с совпадением эмоций
```json
{
  "interval_id": 0,
  "start_ts": "00:00:01.440",
  "person_emotions": {
    "P002": {
      "video_emotion": "neutral",         (видео)
      "audio_sentiment": "positive",      (аудио)
      "combined_emotion": "neutral",      (итого)
      "combined_valence": 0.3058,
      "combined_confidence": 0.8957
    }
  }
}
```

### Общая статистика
```
Total intervals analyzed: 81
Dominant combined emotion: neutral
Average combined valence: 0.3859
Emotion distribution: {
  'neutral': 80,
  'very_positive': 17,
  'positive': 1,
  'negative': 2
}
```

---

## Технические детали ⚙️

### Алгоритм синхронизации
1. Для каждого видео интервала [start, start+10 сек] находятся все аудио сегменты, которые с ним перекрываются
2. Сегменты агрегируются: вычисляется средний sentiment_score
3. Эмоции и сентименты преобразуются в валентность:
   - Видео: neutral=0, happiness=1, sadness=-0.7, anger=-1, и т.д.
   - Аудио: positive=0.7, neutral=0, negative=-0.7
4. **Взвешенное объединение**:
   ```
   combined_valence = (video_valence × video_weight + audio_valence × audio_weight) / total_weight
   ```
5. Валентность преобразуется обратно в категорию эмоции

### Обработка данных
- Входные файлы загружаются из JSON
- Результаты сохраняются в JSON с pretty-print (для читаемости)
- Числовые значения округляются до 4 знаков после запятой

---

## Интеграция в существующий код ✨

**Если у вас уже есть существующих данные анализа:**
1. Эмоции от видео в `data/v3_dense/analysis_step6_emotion_report.json`
2. Сентимент аудио в `data/video_from_bucket_audio_diarized_sentiment.json`

**Просто запустите:**
```bash
python -m app.run_combined_emotion_analysis
```

Выходной файл будет создан в `data/v3_dense/combined_emotion_analysis.json`

---

## Расширение функционала 🔧

### Изменение весов для разных сценариев:
```bash
# Видео-ориентированный анализ (интервью, дебаты)
python -m app.run_combined_emotion_analysis ... 0.7 0.3

# Аудио-ориентированный анализ (лекция, презентация)
python -m app.run_combined_emotion_analysis ... 0.3 0.7
```

### Кастомизация маппинга эмоций:
Отредактируйте `emotion_valence_map` и `sentiment_valence_map` в `combined_emotion_analysis.py` для своих требований.

---

## Структура файлов проекта 📁

```
app/
└── backend/
    └── combined_emotion_analysis/        ← НОВЫЙ МОДУЛЬ
        ├── __init__.py
        ├── combined_emotion_analysis.py  ← Основной код
        └── README.md                     ← Подробная документация

app/
├── run_full_pipeline.py                  ← ОБНОВЛЁН
└── run_combined_emotion_analysis.py      ← НОВЫЙ скрипт для запуска

EXAMPLES_COMBINED_EMOTION.md              ← НОВЫЙ (примеры использования)
```

---

## Проверка работы ✓

Анализ был протестирован на существующих данных:
- ✅ Загрузка эмоций видео: 81 интервал
- ✅ Загрузка сентимента аудио: множество сегментов
- ✅ Синхронизация временных интервалов
- ✅ Вычисление комбинированных эмоций
- ✅ Сохранение результатов в JSON

**Результат:** `data/v3_dense/combined_emotion_analysis.json` (создан успешно)

---

Дата создания: 2026-03-05
Статус: **Production-ready** ✅
