# ==========================================
# ГЛАВНЫЙ КОНВЕЙЕР (pipeline.py)
# Управляющий скрипт, объединяющий все этапы работы.
# ==========================================

import os
import sys

# Импортируем конфигурационный файл
import config

# Импортируем наши независимые логические модули
from trainer import ModelTrainer
from extractor import DataExtractor
from evaluator import ModelEvaluator

class DataPipeline:
    def __init__(self):
        """Инициализация конвейера и проверка доступов."""
        print("=== ИНИЦИАЛИЗАЦИЯ КОНВЕЙЕРА ===")
        if not config.api_key or config.api_key == "вставьте_ваш_api_ключ_сюда":
            print("Критическая ошибка: API-ключ не задан в конфигурации!")
            sys.exit(1)

    def run(self):
        """Строгая последовательность запуска этапов."""
        
        # ЭТАП 1: Обучение модели
        # Защита от повторного обучения: проверяем наличие готовой модели
        if not os.path.exists(config.file_paths["optimized_model"]):
            print("\n[ЭТАП 1] Запуск алгоритмической оптимизации на эталоне (golden)...")
            trainer = ModelTrainer()
            trainer.run_optimization()
        else:
            print(f"\n[ЭТАП 1] Пропущен. Обученная модель уже существует: {config.file_paths['optimized_model']}")

        # ЭТАП 2: Массовое извлечение данных
        print("\n[ЭТАП 2] Запуск чистовой обработки основной базы данных (Baza)...")
        extractor = DataExtractor()
        extractor.run_extraction()

        # ЭТАП 3: Независимая оценка качества (Валидация)
        print("\n[ЭТАП 3] Запуск проверки на экзаменационной выборке (exam)...")
        evaluator = ModelEvaluator()
        evaluator.run_evaluation()

        print("\n=== КОНВЕЙЕР УСПЕШНО ЗАВЕРШИЛ ВСЕ ЗАДАЧИ ===")

if __name__ == "__main__":
    # Единая точка входа. Запускать нужно только этот файл.
    pipeline = DataPipeline()
    pipeline.run()