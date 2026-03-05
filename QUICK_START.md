# Совместный анализ видео+аудио - Быстрый старт ⚡

## Что это?
Новый этап в пайплайне, который комбинирует:
- **Видео анализ** (эмоции из мимики 😊)
- **Аудио анализ** (сентимент из речи 🗣️)

Для получения итогового вывода об эмоциональном состоянии.

---

## 3 способа использования

### 1️⃣ Запустить вместе со всем пайплайном (РЕКОМЕНДУЕТСЯ)
```bash
cd /Users/ekaterinavoronina/CDEK\ uni/cdek

# С вашим видео
python -m app.run_full_pipeline data/my_video.webm

# Или использовать локальное видео
python -m app.run_full_pipeline
```

✅ Новый анализ запустится автоматически в конце
✅ Результат: `data/v3_dense/combined_emotion_analysis.json`

---

### 2️⃣ Запустить только совместный анализ
```bash
python -m app.run_combined_emotion_analysis
```

Будут использованы существующие файлы:
- `data/v3_dense/analysis_step6_emotion_report.json` (эмоции видео)
- `data/video_from_bucket_audio_diarized_sentiment.json` (сентимент аудио)

---

### 3️⃣ Использовать в Python коде
```python
from app.backend.combined_emotion_analysis import run_combined_emotion_analysis

result = run_combined_emotion_analysis(
    emotion_report_path="data/v3_dense/analysis_step6_emotion_report.json",
    audio_sentiment_path="data/video_from_bucket_audio_diarized_sentiment.json",
    output_path="data/v3_dense/combined_emotion_analysis.json"
)

print(f"Основная эмоция: {result['summary']['dominant_combined_emotion']}")
print(f"Средняя валентность: {result['summary']['average_combined_valence']}")
```

---

## Что я получу? 📊

**JSON файл с:**

1. **Общая статистика** (summary)
   ```
   - Доминирующая эмоция: positive/neutral/negative/...
   - Средняя валентность: -1 (очень негативно) до +1 (очень позитивно)
   - Распределение эмоций по интервалам
   ```

2. **Анализ каждого интервала** (combined_intervals)
   ```
   - Видео эмоция (из мимики)
   - Аудио сентимент (из речи)
   - Итоговая эмоция (комбинированная)
   - Доверительность (confidence)
   ```

### Пример результата:
```json
{
  "summary": {
    "total_intervals": 81,
    "dominant_combined_emotion": "neutral",
    "average_combined_valence": 0.3859,
    "emotion_distribution": {
      "neutral": 80,
      "very_positive": 17,
      "positive": 1,
      "negative": 2
    }
  }
}
```

---

## Параметры (опционально) 🎚️

Можно менять веса видео vs аудио:

```bash
python -m app.run_combined_emotion_analysis \
  emotion_path audio_path output_path \
  video_weight audio_weight
```

**Примеры:**
- `0.6 0.4` - видео 60% (мимика важнее) ← **по умолчанию**
- `0.5 0.5` - равный вес
- `0.4 0.6` - аудио 60% (речь важнее)

---

## Категории эмоций 🎭

| Эмоция | Валентность | Пример |
|--------|-------------|---------|
| very_positive | >0.8 | 😄 Очень рад |
| positive | 0.4-0.8 | 🙂 Рад |
| neutral | -0.2-0.4 | 😐 Нейтрально |
| negative | -0.6--0.2 | ☹️ Грустно |
| very_negative | <-0.6 | 😠 Очень зол |

---

## Файлы, которые были добавлены ✨

```
📁 app/backend/combined_emotion_analysis/
   ├── __init__.py
   ├── combined_emotion_analysis.py  (основной код)
   └── README.md                     (подробная документация)

📄 app/run_combined_emotion_analysis.py (скрипт для запуска)

📄 FEATURE_SUMMARY.md (этот отчет)
📄 EXAMPLES_COMBINED_EMOTION.md (примеры кода)
```

---

## Где искать результаты? 📂

После запуска проверьте:

```bash
# Интегрированный анализ
cat data/v3_dense/combined_emotion_analysis.json

# Просмотрите статистику
python -c "
import json
data = json.load(open('data/v3_dense/combined_emotion_analysis.json'))
print(f\"Интервалов: {data['summary']['total_intervals']}\")
print(f\"Основная эмоция: {data['summary']['dominant_combined_emotion']}\")
"
```

---

## Часто задаваемые вопросы 🤔

**Q: Почему видео и аудио разные?**
A: Нормально! Человек может улыбаться (видео) но говорить грустно (аудио). Анализ показывает обе стороны.

**Q: Какой вес использовать?**
A: Зависит от контекста:
- Интервью/переговоры: 0.6-0.7 видео (мимика важна)
- Презентация: 0.3-0.4 видео (слова важнее жестов)

**Q: Можно ли изменить алгоритм?**
A: Да! Отредактируйте `combined_emotion_analysis.py`, функции `emotion_valence_map` и алгоритмы в `_compute_combined_emotion_score()`.

---

## Ошибки и их решение 🔧

### Ошибка: "Error: audio sentiment file not found"
→ Проверьте наличие файла `data/video_from_bucket_audio_diarized_sentiment.json`

### Ошибка: "ImportError: cannot import"
→ Убедитесь, что вы в правильной директории:
```bash
cd /Users/ekaterinavoronina/CDEK\ uni/cdek
```

### Ошибка при запуске всего пайплайна after "TEXT ANALYSIS"
→ Это ожидаемо - проверьте логи выше, может быть проблема с зависимостями Whisper или diarization

---

## Дополнительная информация 📚

- 📖 Полная документация: `app/backend/combined_emotion_analysis/README.md`
- 💻 Примеры кода: `EXAMPLES_COMBINED_EMOTION.md`
- 📝 Этот файл: `QUICK_START.md`

---

**Готово к использованию!** ✅

Вопросы? Смотрите подробные документы или примеры кода.

---

Дата: 2026-03-05
