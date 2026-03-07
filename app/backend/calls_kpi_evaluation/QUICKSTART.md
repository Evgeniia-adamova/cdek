# БЫСТРЫЙ СТАРТ: Анализ звонков по KPI

## 📦 Что было создано

Новый модуль `app/backend/calls_kpi_evaluation/` с полным функционалом для анализа:

```
app/backend/calls_kpi_evaluation/
├── __init__.py                 # Экспорт главных классов
├── kpi_analyzer.py            # Главный анализатор (800+ строк)
├── pipeline_integration.py     # Интеграция в основной pipeline
├── example_usage.py            # Примеры использования
├── test_kpi.py                 # Тесты и примеры
└── README.md                   # Документация
```

## 🚀 Быстрое использование

### Вариант 1: Простой анализ

```python
from app.backend.calls_kpi_evaluation import analyze_call

# Анализируем транскрипцию
results = analyze_call("Менеджер: Здравствуйте... Клиент: Привет...")

# Получаем оценку
print(f"Баллы: {results['overall_score']}/{results['max_possible_score']}")
print(f"Процент: {results['score_percentage']}%")
print(f"Статус: {results['overall_status']}")
```

### Вариант 2: Использование в pipeline

```python
from app.backend.calls_kpi_evaluation.pipeline_integration import add_kpi_to_pipeline_step4

# После извлечения текста
kpi_results = add_kpi_to_pipeline_step4(transcript, video_id)
# Результаты автоматически сохраняются в data/kpi_analysis/
```

### Вариант 3: Тестирование

```bash
cd /Users/ekaterinavoronina/CDEK\ uni/cdek
python -m app.backend.calls_kpi_evaluation.test_kpi
```

## 📊 Что анализирует модуль

**40+ критериев оценки в 5 категориях:**

1. **Установление контакта** (вес 15%)
   - Приветствие
   - Представление менеджера
   - Уточнение имени клиента
   - Обращение по имени

2. **Управление встречей** (вес 20%)
   - Объяснение цели встречи
   - План встречи
   - Подведение итогов
   - Приглашение к вопросам

3. **Сбор информации** (вес 25%)
   - Уточнение объёмов
   - Тип грузов
   - Опыт с конкурентами
   - Приоритеты клиента

4. **Демонстрация услуг** (вес 30%)
   - Стоимость и параметры
   - Упаковка, доп. услуги
   - Страхование
   - Наложенный платёж
   - Платные/бесплатные каналы
   - Спец. услуги (фулфилмент, реверс, возврат)
   - Интеграция

5. **Финализация** (вес 10%)
   - Предупреждение о задержках
   - Отправка материалов
   - Запрос обратной связи

## 📈 Результаты анализа

Каждый звонок получает:

- **Общую оценку**: X/100 баллов
- **Процент выполнения**: 0-100%
- **Статус**: 
  - ★★★★★ Отлично (91-100%)
  - ★★★★ Хорошо (76-90%)
  - ★★★ Удовлетворительно (51-75%)
  - ★★ Требует развития (0-50%)
- **Детальный отчет по каждому критерию** с обоснованием

## 💾 Где сохраняются результаты

```
data/kpi_analysis/
├── kpi_analysis_{video_id}.json
└── ...
```

Каждый файл содержит полный отчет в JSON формате.

## 🔌 Интеграция в основной pipeline

Добавить в `app/run_full_pipeline.py` после step 3:

```python
from app.backend.calls_kpi_evaluation.pipeline_integration import add_kpi_to_pipeline_step4

# После combined_emotion_analysis
kpi_results = add_kpi_to_pipeline_step4(transcript, video_id)

# Объединить результаты
final_results = {
    'emotion': combined_emotion_result,
    'kpi': kpi_results
}
```

## 📝 Примеры результатов

**Хороший звонок (80+ баллов):**
```json
{
  "overall_score": 75.5,
  "score_percentage": 75.5,
  "overall_status": "Хорошо",
  "rating": "★★★★",
  "categories": {
    "Установление контакта": {
      "total_score": 8,
      "max_score": 8,
      "percentage": 100%
    },
    ...
  }
}
```

## 🛠️ Возможные улучшения в будущем

1. NLP анализ для более точных проверок
2. ML классификатор для сложных критериев
3. Анализ эмоционального контекста
4. Временной анализ (сколько времени на каждый этап)
5. Сравнение с эталонными диалогами

## 📞 Использование примеров

```bash
# Запустить тесты
python app/backend/calls_kpi_evaluation/test_kpi.py

# Или ручной анализ
python app/backend/calls_kpi_evaluation/example_usage.py

# Или в интерпретаторе:
python -c "from app.backend.calls_kpi_evaluation import analyze_call; print(analyze_call('текст'))"
```

## ✅ Что можно делать сейчас

1. ✓ Анализировать любую транскрипцию в коде
2. ✓ Сохранять результаты в JSON
3. ✓ Интегрировать в основной pipeline
4. ✓ Выводить красивые отчеты
5. ✓ Обрабатывать пакеты звонков
6. ✓ Объединять с анализом эмоций

## 📞 Как начать использовать прямо сейчас

```python
# Самый простой способ - в терминале Python:
from app.backend.calls_kpi_evaluation import analyze_call

results = analyze_call("Менеджер: Здравствуйте, меня зовут Иван...")
print(results['overall_score'])
print(results['overall_status'])
```

Готово! Модуль полностью функционален и готов к использованию.
