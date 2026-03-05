# 📚 Документация: Совместный анализ видео+аудио эмоций

## Содержание

### 1. **QUICK_START.md** ⚡ (Начните отсюда!)
   - 3 способа использования
   - Основные параметры
   - Категории эмоций
   - FAQ
   
   👉 **Для быстрого старта**: читайте этот файл первым

### 2. **FEATURE_SUMMARY.md** 📊
   - Полный отчет о реализации
   - Что было добавлено
   - Структура выходных данных
   - Примеры результатов
   - Проверка работы
   
   👉 **Для полного понимания**: читайте после Quick Start

### 3. **EXAMPLES_COMBINED_EMOTION.md** 💻
   - 8 практических примеров (от простых к сложным)
   - Запуск через пайплайн
   - Запуск отдельно
   - Программное использование в Python
   - Экспериментирование с весами
   - Анализ результатов
   - Визуализация и экспорт
   - Диагностика и отладка
   
   👉 **Для практической реализации**: используйте эти примеры

### 4. **ARCHITECTURE.md** 🏗️
   - ASCII диаграмма архитектуры
   - Компоненты системы
   - Потоки данных
   - Параметры и конфигурация
   - Use cases
   
   👉 **Для понимания дизайна**: смотрите диаграммы

### 5. **app/backend/combined_emotion_analysis/README.md** 📖
   - Описание модуля
   - API функций
   - Структура выходных данных
   - Алгоритм синхронизации
   - Расширение функционала
   
   👉 **Для технических деталей**:읽read модуль README

---

## 🚀 Как начать?

### Вариант A: Я хочу быстро запустить (5 минут)
1. Прочитайте **QUICK_START.md**
2. Выполните одну из 3 команд:
   ```bash
   # Вариант 1: Весь пайплайн
   python -m app.run_full_pipeline data/my_video.webm
   
   # Вариант 2: Только совместный анализ
   python -m app.run_combined_emotion_analysis
   
   # Вариант 3: В коде Python
   from app.backend.combined_emotion_analysis import run_combined_emotion_analysis
   ```

### Вариант B: Я хочу понять, как это работает (30 минут)
1. Прочитайте **QUICK_START.md** (5 мин)
2. Прочитайте **ARCHITECTURE.md** (10 мин)
3. Обратитесь к **EXAMPLES_COMBINED_EMOTION.md** (15 мин)

### Вариант C: Я хочу всё знать (1 час)
1. QUICK_START.md (5 мин)
2. FEATURE_SUMMARY.md (10 мин)
3. ARCHITECTURE.md (15 мин)
4. EXAMPLES_COMBINED_EMOTION.md (20 мин)
5. app/backend/combined_emotion_analysis/README.md (10 мин)

---

## 📁 Структура файлов

```
/Users/ekaterinavoronina/CDEK\ uni/cdek/
├── QUICK_START.md                    ← Начните с этого! ⚡
├── FEATURE_SUMMARY.md                ← Полный отчет 📊
├── EXAMPLES_COMBINED_EMOTION.md      ← Примеры кода 💻
├── ARCHITECTURE.md                   ← Архитектура 🏗️
├── INDEX.md                          ← Этот файл 📚
│
├── app/
│   ├── run_full_pipeline.py
│   ├── run_combined_emotion_analysis.py  ← НОВЫЙ скрипт для запуска
│   │
│   └── backend/
│       └── combined_emotion_analysis/    ← НОВЫЙ модуль
│           ├── __init__.py
│           ├── combined_emotion_analysis.py
│           └── README.md
│
└── data/
    └── v3_dense/
        └── combined_emotion_analysis.json  ← Выходной файл
```

---

## 🎯 Основные концепции

### Что такое "combined emotion"?
Это результат объединения:
- **Видео** (мимика лица) → эмоции: happiness, sadness, neutral, anger, fear, disgust, surprise, contempt
- **Аудио** (речь) → сентимент: positive, neutral, negative

В единую шкалу **валентность** [-1, +1]:
- **+1**: очень позитивная эмоция
- **0**: нейтральная эмоция  
- **-1**: очень негативная эмоция

### Как это работает?
1. **Синхронизация**: для каждого видео интервала (10 сек) находятся аудио сегменты, которые с ним совпадают по времени
2. **Преобразование**: каждая эмоция и сентимент преобразуются в число [-1, +1]
3. **Комбинирование**: результаты объединяются с использованием весов (по умолчанию видео 60%, аудио 40%)
4. **Категоризация**: валентность преобразуется обратно в категорию эмоции

### Использование весов
- **Video-focused** (0.7 видео, 0.3 аудио): интервью, дебаты → мимика лица более выразительна
- **Balanced** (0.6 видео, 0.4 аудио): обычный случай **← по умолчанию**
- **Audio-focused** (0.3 видео, 0.7 аудио): лекция, презентация → слова более важны

---

## 💡 Быстрые ответы на вопросы

### "Как запустить анализ?"
**Быстро**: 
```bash
python -m app.run_combined_emotion_analysis
```

**Полностью**:
```bash
python -m app.run_full_pipeline data/my_video.webm
```

→ Читай **QUICK_START.md**

### "Где найти результаты?"
`data/v3_dense/combined_emotion_analysis.json`

→ Читай **FEATURE_SUMMARY.md** → раздел "Выходные данные"

### "Как изменить веса?"
```bash
python -m app.run_combined_emotion_analysis ... 0.4 0.6
```

→ Читай **QUICK_START.md** → раздел "Параметры"

### "Как использовать в коде?"
```python
from app.backend.combined_emotion_analysis import run_combined_emotion_analysis
result = run_combined_emotion_analysis(...)
```

→ Читай **EXAMPLES_COMBINED_EMOTION.md** → раздел "Программное использование"

### "Как анализировать результаты?"
Несколько примеров в **EXAMPLES_COMBINED_EMOTION.md** → раздел "Анализ результатов в Python"

---

## 🔗 Связанные файлы

- **Видео анализ (Step 6)**: `analysis_step6_emotion_report.json` (видная эмоция от мимики)
- **Аудио анализ**: `video_from_bucket_audio_diarized_sentiment.json` (сентимент речи)
- **Совместный анализ**: `combined_emotion_analysis.json` (итоговый результат)

---

## ✅ Checklist использования

- [ ] Прочитал QUICK_START.md
- [ ] Запустил `python -m app.run_combined_emotion_analysis` (или другой вариант)
- [ ] Проверил результаты в `data/v3_dense/combined_emotion_analysis.json`
- [ ] Понял структуру выходных данных
- [ ] Прочитал EXAMPLES_COMBINED_EMOTION.md для практики
- [ ] Адаптировал для своего use case

---

## 🐛 Нужна помощь?

### Проблема при запуске?
→ Смотрите **QUICK_START.md** → раздел "Ошибки и их решение"

### Не понимаю результаты?
→ Читайте **FEATURE_SUMMARY.md** → раздел "Выходные данные"

### Хочу кастомизировать?
→ Смотрите **app/backend/combined_emotion_analysis/README.md** → раздел "Расширение функционала"

### Нужен пример похожего кода?
→ Обратитесь к **EXAMPLES_COMBINED_EMOTION.md** (8 полных примеров)

---

## 📊 Статистика реализации

| Компонент | Файл | Статус |
|-----------|------|--------|
| Основной модуль | `combined_emotion_analysis.py` | ✅ Production-ready |
| Скрипт запуска | `run_combined_emotion_analysis.py` | ✅ Работает |
| Интеграция в пайплайн | `run_full_pipeline.py` | ✅ Добавлена |
| Документация | Этот INDEX.md + README.md | ✅ Полная |
| Примеры | EXAMPLES_COMBINED_EMOTION.md | ✅ 8 примеров |
| Архитектура | ARCHITECTURE.md | ✅ ASCII диаграммы |
| Feature Summary | FEATURE_SUMMARY.md | ✅ Полный отчет |
| Quick Start | QUICK_START.md | ✅ Готово к использованию |

**Общая готовность**: 🟢 **100%**

---

**Дата последнего обновления**: 2026-03-05
**Статус**: Production-Ready ✅

**Рекомендация**: Начните с **QUICK_START.md** → потом обратитесь к другим файлам по мере необходимости.
