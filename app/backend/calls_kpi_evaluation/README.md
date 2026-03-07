# KPI Analysis for Call Evaluation
# Анализ звонков по KPI критериям

Модуль для автоматического анализа транскрипций звонков сотрудников по установленным KPI критериям.

## Особенности

- **40+ критериев оценки**: Полное покрытие всех этапов продажного диалога
- **5 категорий анализа**: Установление контакта, управление встречей, сбор информации, демонстрация услуг, финализация
- **Итоговая оценка**: Автоматический расчет баллов и процента выполнения KPI
- **Интеграция в pipeline**: Готовые функции для встраивания в основной workflow
- **JSON результаты**: Все результаты сохраняются в машиночитаемом формате

## Структура оценки

### 1. Установление контакта (вес: 15%)
- Приветствие клиента (2 балла)
- Представление менеджера (2 балла)
- Уточнение имени клиента (2 балла)
- Обращение по имени (2 балла)

### 2. Управление встречей (вес: 20%)
- Объяснение цели встречи (2 балла)
- План встречи (2 балла)
- Подведение итогов (5 баллов)
- Приглашение к вопросам (4 балла)

### 3. Сбор информации (вес: 25%)
- Уточнение объёмов (3 балла)
- Тип грузов (3 балла)
- Опыт с конкурентами (2 балла)
- Приоритеты и предпочтения (2 балла)

### 4. Демонстрация услуг (вес: 30%)
- Стоимость и параметры груза (1 балл)
- Оптовая упаковка (2 балла)
- Дополнительные услуги (2 балла)
- Страхование (5 баллов)
- Наложенный платёж и инкассация (5 баллов)
- Платные каналы связи (5 баллов)
- Бесплатный канал - почта (10 баллов)
- Процесс оплаты (5 баллов)
- Окончание депозита (5 баллов)
- Специальные услуги - фулфилмент, реверс, клиентский возврат (15 баллов)
- Интеграция (5 баллов)
- Взаимодействие с компанией (1 балл)

### 5. Финализация (вес: 10%)
- Предупреждение о задержках ВА (3 балла)
- Отправка материалов (2 балла)
- Запрос обратной связи (3 балла)

## Использование

### Простой анализ

```python
from app.backend.calls_kpi_evaluation import analyze_call

# Анализируем транскрипцию
results = analyze_call("Менеджер: Здравствуйте... Клиент: Привет...")

# Получаем результаты
print(f"Оценка: {results['overall_score']}/{results['max_possible_score']}")
print(f"Процент: {results['score_percentage']}%")
print(f"Статус: {results['overall_status']}")
```

### Интеграция в pipeline

```python
from app.backend.calls_kpi_evaluation.pipeline_integration import add_kpi_to_pipeline_step4

# После step 3 (текст извлечен)
kpi_results = add_kpi_to_pipeline_step4(transcript, video_id)

# Результаты сохраняются автоматически в data/kpi_analysis/
```

### Анализ JSON файла

```python
from app.backend.calls_kpi_evaluation.example_usage import analyze_from_json

results = analyze_from_json("data/v3_dense/combined_emotion_analysis.json")
```

### Пакетная обработка

```python
from app.backend.calls_kpi_evaluation.pipeline_integration import KPIPipelineIntegration

integration = KPIPipelineIntegration()
batch_results = integration.process_batch("data/v3_dense")

print(f"Обработано: {batch_results['processed']}")
print(f"Средняя оценка: {batch_results['average_score']}%")
```

## Структура результатов

```json
{
  "overall_score": 65.5,
  "max_possible_score": 100,
  "score_percentage": 65.5,
  "overall_status": "Удовлетворительно",
  "rating": "★★★",
  "categories": {
    "Установление контакта": {
      "total_score": 8,
      "max_score": 8,
      "percentage": 100.0,
      "items": [...]
    },
    ...
  },
  "all_items": [...]
}
```

## Интеграция в основной pipeline

### Вариант 1: Добавить в run_full_pipeline.py

```python
from app.backend.calls_kpi_evaluation.pipeline_integration import add_kpi_to_pipeline_step4

# After step 3 (combined_emotion_analysis)
kpi_results = add_kpi_to_pipeline_step4(transcript, video_id)

# Объединяем с остальными результатами
final_results = {
    'emotion': combined_emotion_result,
    'kpi': kpi_results
}
```

### Вариант 2: Автономный анализ

```bash
python -m app.backend.calls_kpi_evaluation.example_usage
```

## Результаты

Все результаты сохраняются в `data/kpi_analysis/` в формате JSON:
- Файл: `kpi_analysis_{video_id}.json`
- Содержит: полный отчет с оценками по каждому критерию

## Примечания

- Анализ основан на регулярных выражениях и поиске ключевых фраз
- Для более точного анализа рекомендуется использовать вместе с NLP моделями
- Чувствительность к опечаткам в транскрипции может быть улучшена
- Рекомендуется использовать вывод высокого качества из модели транскрипции

## Возможные улучшения

1. Интеграция с NER/NLP для более точного выявления критериев
2. Анализ эмационального контекста при оценке критериев
3. Взвешивание критериев по важности
4. Machine learning классификатор для сложных критериев
5. Анализ времени на каждый этап диалога

## Разработчик

Создано для CDEK - система анализа качества звонков сотрудников
