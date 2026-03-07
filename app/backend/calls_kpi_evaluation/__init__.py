"""
Calls KPI Evaluation Module
Анализ звонков сотрудников по установленным KPI критериям
"""

from .kpi_analyzer import CallKPIAnalyzer, analyze_call

__all__ = ['CallKPIAnalyzer', 'analyze_call']
