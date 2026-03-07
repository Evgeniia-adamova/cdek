# 🎯 KPI АНАЛИЗА ЗВОНКОВ - ШПАРГАЛКА

## ✅ Быстрая проверка что все работает

```bash
# Основная команда проверки
python verify_kpi_module.py

# Или через bash скрипт
./kpi.sh verify
```

## 🚀 Быстрый старт

### Вариант 1: Интерактивный анализ (самый простой)
```bash
python run_kpi_analyzer.py
```
Выберите опцию, вставьте текст и получите результат.

### Вариант 2: Bash команды (удобно)
```bash
./kpi.sh verify        # Проверить что работает
./kpi.sh test          # Запустить тесты
./kpi.sh run           # Интерактивный анализатор
./kpi.sh example       # Пример использования
./kpi.sh batch         # Обработать пакет файлов
./kpi.sh results       # Просмотреть результаты
./kpi.sh help          # Справка
```

### Вариант 3: Python код (для интеграции)
```python
from app.backend.calls_kpi_evaluation import analyze_call

# Анализируем транскрипцию
results = analyze_call("Менеджер: Здравствуйте... Клиент: Привет...")

# Смотрим результат
print(f"Оценка: {results['overall_score']}/100")
print(f"Статус: {results['overall_status']} {results['rating']}")
```

### Вариант 4: Интеграция в pipeline
```python
from app.backend.calls_kpi_evaluation.pipeline_integration import add_kpi_to_pipeline_step4

# После извлечения текста
kpi_results = add_kpi_to_pipeline_step4(transcript, video_id)
# Результаты сохранены в data/kpi_analysis/
```

## 📊 Что анализируется

| Критерий | Баллы | Проверяет |
|----------|-------|-----------|
| **Установление контакта** | 8 | Приветствие, представление, имя |
| **Управление встречей** | 13 | Цель, план, итоги, вопросы |
| **Сбор информации** | 10 | Объёмы, товары, конкуренты |
| **Демонстрация услуг** | 61 | Услуги, платежи, интеграция |
| **Финализация** | 8 | Материалы, обратная связь |
| **ИТОГО** | **100** | |

## 📁 Где сохраняются результаты

```
data/kpi_analysis/
├── kpi_analysis_video_1.json
├── kpi_analysis_video_2.json
└── ...
```

Каждый файл содержит:
- Общую оценку (0-100 баллов)
- Проценты по категориям
- Детальный отчет по каждому критерию

## 📖 Документация

| Файл | Описание |
|------|---------|
| `KPI_ANALYSIS_FINAL_DOCUMENTATION.md` | Полное руководство |
| `app/backend/calls_kpi_evaluation/README.md` | Техническая документация |
| `app/backend/calls_kpi_evaluation/QUICKSTART.md` | Быстрый старт |

## 🧪 Тестирование

```bash
# Запустить встроенные тесты
./kpi.sh test
# или
python app/backend/calls_kpi_evaluation/test_kpi.py
```

**Результаты тестов:**
- Хороший звонок: 68/100 ✓
- Плохой звонок: 8/100 ✓

## 🔧 Проверка что нужно

```bash
# Убедиться что все файлы на месте
./kpi.sh verify

# Вывод:
# ✓ Файлы
# ✓ Импорты
# ✓ Функциональность
# ✓ Директория результатов
# ✓ Тестовые результаты
# ✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!
```

## 💡 Примеры

### Анализ JSON файла
```python
from app.backend.calls_kpi_evaluation.example_usage import analyze_from_json

results = analyze_from_json("data/v3_dense/combined_emotion_analysis.json")
print(results['overall_score'])
```

### Обработка пакета файлов
```python
from app.backend.calls_kpi_evaluation.pipeline_integration import KPIPipelineIntegration

integration = KPIPipelineIntegration()
batch = integration.process_batch("data/v3_dense")

print(f"Обработано: {batch['processed']}")
print(f"Средняя оценка: {batch['average_score']}%")
```

### Вывод красивого отчета
```python
from app.backend.calls_kpi_evaluation.example_usage import print_analysis_report

results = analyze_call(transcript)
print_analysis_report(results)
```

## 📊 Шкала оценок

| Процент | Рейтинг | Статус |
|---------|---------|--------|
| 91-100% | ★★★★★ | Отлично |
| 76-90% | ★★★★ | Хорошо |
| 51-75% | ★★★ | Удовлетворительно |
| 0-50% | ★★ | Требует развития |

## ❓ Часто задаваемые вопросы

**Q: Как использовать в своем коде?**
```python
from app.backend.calls_kpi_evaluation import analyze_call
results = analyze_call("ваш текст")
```

**Q: Где результаты?**
```bash
ls -la data/kpi_analysis/
```

**Q: Как интегрировать в pipeline?**
Добавить в `app/run_full_pipeline.py`:
```python
from app.backend.calls_kpi_evaluation.pipeline_integration import add_kpi_to_pipeline_step4
kpi_results = add_kpi_to_pipeline_step4(transcript, video_id)
```

**Q: Как запустить только тесты?**
```bash
./kpi.sh test
```

**Q: Как проверить что все работает?**
```bash
./kpi.sh verify
```

## 🎯 Команда в одну строку

```bash
# Проверить и запустить
python verify_kpi_module.py && python app/backend/calls_kpi_evaluation/test_kpi.py
```

---

**Готово! Модуль полностью функционален и готов к использованию.** 🚀
