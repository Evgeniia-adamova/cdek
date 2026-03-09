"""
Calls KPI Evaluation Module — СДЭК Чеклист
Анализ звонков по 29 критериям, суммарно 100 баллов.
"""

from .kpi_analyzer import CallKPIAnalyzer, analyze_call, CHECKLIST_CRITERIA
from .pipeline_integration import run_kpi_evaluation, run_kpi_from_file, print_kpi_report

__all__ = [
    'CallKPIAnalyzer',
    'analyze_call',
    'CHECKLIST_CRITERIA',
    'run_kpi_evaluation',
    'run_kpi_from_file',
    'print_kpi_report',
]
