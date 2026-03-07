# 🚀 КОМАНДЫ ДЛЯ ЗАПУСКА KPI АНАЛИЗА

## ⚡ 3 ОСНОВНЫЕ КОМАНДЫ

### 1️⃣ Интерактивный анализ (САМЫЙ ПРОСТОЙ)
```bash
python run_kpi_analyzer.py
```
**Что делает?** Запускает интерактивное меню где вы можете:
- Вставить текст транскрипции вручную
- Загрузить транскрипцию из файла
- Запустить встроенные тесты

---

### 2️⃣ Быстрая проверка что всё работает
```bash
python verify_kpi_module.py
```
**Что делает?** Проверяет:
- ✓ Все файлы на месте
- ✓ Импорты работают
- ✓ Функциональность корректна
- ✓ Директория для результатов готова

---

### 3️⃣ Запустить тесты
```bash
./kpi.sh test
```
**Что делает?** Анализирует примеры хорошего и плохого звонка:
- Хороший звонок: 68/100 ✓
- Плохой звонок: 8/100 ✓

---

## 📝 ОСТАЛЬНЫЕ КОМАНДЫ

### Быстрые команды через bash:
```bash
./kpi.sh verify       # Проверить компоненты
./kpi.sh test         # Запустить тесты
./kpi.sh run          # Интерактивный анализатор
./kpi.sh example      # Пример из JSON
./kpi.sh batch        # Обработать пакет файлов
./kpi.sh results      # Просмотреть результаты
./kpi.sh help         # Справка
```

### Специализированные команды:
```bash
# Примеры использования
python app/backend/calls_kpi_evaluation/test_kpi.py
python app/backend/calls_kpi_evaluation/example_usage.py
python app/backend/calls_kpi_evaluation/pipeline_integration.py
```

### Python код (для интеграции):
```bash
python -c "from app.backend.calls_kpi_evaluation import analyze_call; 
           results = analyze_call('Менеджер: Здравствуйте'); 
           print(results['overall_score'])"
```

---

## 📊 ГДЕ РЕЗУЛЬТАТЫ

```bash
# Просмотреть все результаты
ls -la data/kpi_analysis/

# Просмотреть конкретный результат
cat data/kpi_analysis/kpi_analysis_*.json | jq
```

---

## 📖 ДОКУМЕНТАЦИЯ

```bash
# Шпаргалка
cat KPI_QUICK_START.md

# Полная документация
cat KPI_ANALYSIS_FINAL_DOCUMENTATION.md

# README модуля
cat app/backend/calls_kpi_evaluation/README.md

# Все команды
./kpi_commands.sh
```

---

## 🎯 РЕКОМЕНДНЫЙ ПОРЯДОК

1. **Сначала** проверьте что всё работает:
   ```bash
   python verify_kpi_module.py
   ```

2. **Затем** запустите интерактивный анализатор:
   ```bash
   python run_kpi_analyzer.py
   ```

3. **Или** используйте быстрые команды:
   ```bash
   ./kpi.sh test
   ```

---

## ✅ ЧТО АНАЛИЗИРУЕТ

- **40+ KPI критериев** в 5 категориях
- **100 баллов максимум**
- **Рейтинг качества**: ★★★★★ до ★★
- **JSON результаты** с подробным отчетом

---

**Готово! Выберите удобную команду выше и начните анализ! 🚀**
