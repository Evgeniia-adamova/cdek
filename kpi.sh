#!/bin/bash
# KPI Module Quick Commands
# Быстрые команды для работы с модулем KPI анализа

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_command() {
    echo -e "${YELLOW}$${NC} $1"
}

case "${1:-help}" in
    verify|check)
        print_header "Проверка что все работает"
        python verify_kpi_module.py
        ;;
    
    test)
        print_header "Запуск полных тестов KPI модуля"
        python app/backend/calls_kpi_evaluation/test_kpi.py
        ;;
    
    run|interactive)
        print_header "Запуск интерактивного анализатора"
        python run_kpi_analyzer.py
        ;;
    
    example)
        print_header "Запуск примера использования"
        python app/backend/calls_kpi_evaluation/example_usage.py
        ;;
    
    batch)
        print_header "Обработка пакета файлов"
        print_info "Обрабатываем data/v3_dense"
        python app/backend/calls_kpi_evaluation/pipeline_integration.py
        ;;
    
    results)
        print_header "Просмотр результатов анализа"
        if [ -d "data/kpi_analysis" ]; then
            ls -lh data/kpi_analysis/
            echo ""
            if [ "$(ls -A data/kpi_analysis)" ]; then
                print_success "Найдены результаты анализа:"
                find data/kpi_analysis -name "*.json" | while read file; do
                    echo "  - $(basename $file)"
                done
            else
                print_info "Директория пуста - запустите анализ"
            fi
        else
            print_info "Директория data/kpi_analysis еще не создана"
        fi
        ;;
    
    docs|help)
        print_header "KPI Module - Справка"
        echo ""
        echo "Доступные команды:"
        echo ""
        echo "  kpi.sh verify        - Проверить что все работает"
        echo "  kpi.sh test          - Запустить полные тесты"
        echo "  kpi.sh run           - Интерактивный анализатор"
        echo "  kpi.sh example       - Пример использования"
        echo "  kpi.sh batch         - Обработать пакет файлов"
        echo "  kpi.sh results       - Просмотреть результаты"
        echo "  kpi.sh docs          - Показать эту справку"
        echo ""
        echo "Примеры использования в Python:"
        echo ""
        echo "  from app.backend.calls_kpi_evaluation import analyze_call"
        echo "  results = analyze_call('текст диалога')"
        echo ""
        echo "Документация:"
        echo "  - KPI_ANALYSIS_FINAL_DOCUMENTATION.md (полная инструкция)"
        echo "  - app/backend/calls_kpi_evaluation/README.md"
        echo "  - app/backend/calls_kpi_evaluation/QUICKSTART.md"
        echo ""
        ;;
    
    *)
        echo "❌ Неизвестная команда: $1"
        echo ""
        echo "Используйте: kpi.sh help"
        exit 1
        ;;
esac
