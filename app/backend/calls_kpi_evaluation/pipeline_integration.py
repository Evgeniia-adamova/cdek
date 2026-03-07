"""
KPI Analysis Pipeline Integration
Интеграция анализа KPI в основной pipeline
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from calls_kpi_evaluation import analyze_call


class KPIPipelineIntegration:
    """
    Интеграция анализа KPI в основной pipeline
    Анализирует звонки после извлечения текста
    """
    
    def __init__(self, save_results: bool = True, output_dir: str = None):
        """
        Args:
            save_results: Сохранять ли результаты в файл
            output_dir: Директория для сохранения результатов
        """
        self.save_results = save_results
        self.output_dir = Path(output_dir) if output_dir else Path("data/kpi_analysis")
        if self.save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_and_save(self, transcript: str, video_id: str, metadata: Dict = None) -> Dict:
        """
        Анализирует транскрипцию и сохраняет результаты
        
        Args:
            transcript: Текст транскрипции
            video_id: ID видео/звонка
            metadata: Дополнительные метаданные
            
        Returns:
            Результаты анализа с сохраненным путем
        """
        
        # Анализируем
        raw_results = analyze_call(transcript)
        
        # Преобразуем результаты в удобный формат для pipeline
        category_mapping = {
            "Этап 1: Установление контакта": "greeting_and_politeness",
            "Этап 2: Управление встречей": "meeting_management",
            "Этап 3: Сбор информации": "information_gathering",
            "Этап 4: Демонстрация услуг": "service_demonstration",
            "Этап 5: Финализация": "finalization",
        }
        
        category_scores = {}
        for cat_key, cat_value in raw_results.get('categories', {}).items():
            # Простое преобразование: используем название как ключ
            category_scores[cat_key] = cat_value['total_score']
        
        # Подготавливаем детальные проверки (boolean результаты)
        detailed_checks = {}
        for item in raw_results.get('all_items', []):
            check_name = item.get('key', item.get('name', 'unknown'))
            detailed_checks[check_name] = item.get('status') == 'yes'
        
        # Формируем результат в стандартном формате
        results = {
            'video_id': video_id,
            'timestamp': datetime.now().isoformat(),
            'final_score': raw_results.get('overall_score', 0),
            'grade': f"{raw_results.get('overall_status', 'N/A')} {raw_results.get('rating', '')}",
            'summary': f"Оценка {raw_results.get('overall_status', 'N/A').lower()}: {raw_results.get('score_percentage', 0):.1f}%",
            'category_scores': category_scores,
            'detailed_checks': detailed_checks,
            'raw_results': raw_results,  # Сохраняем полные результаты для анализа
        }
        
        if metadata:
            results['metadata'] = metadata
        
        # Сохраняем результаты
        if self.save_results:
            output_file = self.output_dir / f"kpi_analysis_{video_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            results['saved_to'] = str(output_file)
        
        return results
    
    def merge_with_emotion_analysis(self, emotion_analysis: Dict, kpi_analysis: Dict) -> Dict:
        """
        Объединяет результаты эмоционального анализа с KPI анализом
        
        Args:
            emotion_analysis: Результаты анализа эмоций
            kpi_analysis: Результаты анализа KPI
            
        Returns:
            Объединенный результат
        """
        
        merged = {
            'timestamp': datetime.now().isoformat(),
            'analysis': {
                'emotion': emotion_analysis,
                'kpi': kpi_analysis
            }
        }
        
        # Добавляем общую оценку
        if kpi_analysis.get('score_percentage'):
            merged['overall_kpi_score'] = kpi_analysis['score_percentage']
        
        return merged
    
    def process_batch(self, transcripts_dir: str) -> Dict:
        """
        Обрабатывает пакет транскрипций из директории
        
        Args:
            transcripts_dir: Директория с JSON файлами транскрипций
            
        Returns:
            Статистика обработки
        """
        
        transcripts_path = Path(transcripts_dir)
        results = {
            'processed': 0,
            'errors': 0,
            'total_score': 0,
            'analyses': []
        }
        
        for json_file in transcripts_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Извлекаем текст
                transcript = data.get('text') or data.get('transcript') or ""
                if not transcript:
                    continue
                
                # Анализируем
                analysis = self.analyze_and_save(
                    transcript,
                    json_file.stem,
                    metadata={'source_file': str(json_file)}
                )
                
                results['analyses'].append({
                    'file': json_file.name,
                    'score': analysis['overall_score'],
                    'percentage': analysis['score_percentage'],
                    'status': analysis['overall_status']
                })
                
                results['processed'] += 1
                results['total_score'] += analysis['score_percentage']
                
            except Exception as e:
                results['errors'] += 1
                print(f"Ошибка при обработке {json_file.name}: {str(e)}")
        
        # Средняя оценка
        if results['processed'] > 0:
            results['average_score'] = round(results['total_score'] / results['processed'], 1)
        
        return results


def integrate_with_pipeline(transcript: str, video_id: str, 
                           combined_emotion_result: Optional[Dict] = None) -> Dict:
    """
    Главная функция для интеграции в pipeline
    Вызывается после text_extraction и перед сохранением результатов
    
    Args:
        transcript: Извлеченный текст
        video_id: ID видео
        combined_emotion_result: Результаты combined_emotion_analysis (опционально)
        
    Returns:
        Полный результат анализа с KPI
    """
    
    integration = KPIPipelineIntegration(save_results=True)
    
    # Анализируем KPI
    kpi_analysis = integration.analyze_and_save(transcript, video_id)
    
    # Если есть результаты эмоционального анализа, объединяем
    if combined_emotion_result:
        full_result = integration.merge_with_emotion_analysis(
            combined_emotion_result,
            kpi_analysis
        )
    else:
        full_result = {
            'kpi_analysis': kpi_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    return full_result


# Для использования в run_full_pipeline.py
def add_kpi_to_pipeline_step4(transcript: str, video_id: str) -> Dict:
    """
    Готовая функция для добавления в run_full_pipeline.py
    
    Usage:
        from app.backend.calls_kpi_evaluation.pipeline_integration import add_kpi_to_pipeline_step4
        
        # После step 3 (text_extraction)
        kpi_results = add_kpi_to_pipeline_step4(transcript, video_id)
    """
    
    integration = KPIPipelineIntegration(save_results=True)
    return integration.analyze_and_save(transcript, video_id)


if __name__ == "__main__":
    # Пример: обработка пакета файлов
    integration = KPIPipelineIntegration()
    
    # Обрабатываем данные V3
    batch_results = integration.process_batch("data/v3_dense")
    
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ПАКЕТНОЙ ОБРАБОТКИ")
    print("="*80)
    print(f"Обработано: {batch_results['processed']}")
    print(f"Ошибок: {batch_results['errors']}")
    if batch_results['processed'] > 0:
        print(f"Средняя оценка: {batch_results.get('average_score', 'N/A')}%")
    print("\nДетали:")
    for analysis in batch_results['analyses']:
        print(f"  {analysis['file']}: {analysis['percentage']}% ({analysis['status']})")
