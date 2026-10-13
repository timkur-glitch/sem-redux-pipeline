# ==========================================
# МОДУЛЬ МАССОВОГО ИЗВЛЕЧЕНИЯ (extractor.py)
# Применяет обученную модель к основной базе данных.
# ==========================================

import os
import time
import csv
import pandas as pd
import dspy

# Импортируем конфигурационный файл проекта
import config

# 1. СХЕМА ИЗВЛЕЧЕНИЯ (Должна полностью совпадать со схемой из trainer.py)
class ExtractionSignature(dspy.Signature):
    """Прочитай исходный текст и извлеки целевые характеристики в 1-2 слова."""
    text_input = dspy.InputField(desc="Исходный текст для анализа")
    feature_1 = dspy.OutputField(desc="Первая характеристика (1-2 слова)")
    feature_2 = dspy.OutputField(desc="Вторая характеристика (1-2 слова)")


class DataExtractor:
    def __init__(self):
        """Инициализация языковой модели и загрузка обученного алгоритма."""
        # Подтягиваем доступы и настройки из config.py
        self.language_model = dspy.LM(config.model_name, api_key=config.api_key)
        dspy.settings.configure(lm=self.language_model)
        
        self.predictor = dspy.Predict(ExtractionSignature)
        
        # Загружаем "поумневшую" модель, сохраненную на этапе обучения
        model_path = config.file_paths["optimized_model"]
        if os.path.exists(model_path):
            self.predictor.load(model_path)
            print(f"Обученная модель успешно загружена из {model_path}")
        else:
            raise FileNotFoundError("Обученная модель не найдена. Сначала запустите модуль trainer.py")

    def _setup_output_file(self, total_rows):
        """Создает выходной файл с заголовками, если он еще не существует."""
        # Генерируем динамическое имя файла на основе размера базы
        output_file = os.path.join(
            config.file_paths["output_folder"], 
            f"{total_rows}_results_final.csv"
        )
        
        if not os.path.exists(output_file):
            os.makedirs(config.file_paths["output_folder"], exist_ok=True)
            with open(output_file, mode='w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                # Пользователь должен указать здесь свои целевые столбцы
                writer.writerow(["Идентификатор_документа", "Текст", "Признак_1", "Признак_2"])
        
        return output_file

    def run_extraction(self):
        """Запускает цикл обработки всей базы данных с защитой от сбоев."""
        print("Загрузка исходной базы данных...")
        df = pd.read_csv(config.file_paths["raw_database"], sep=';', encoding='utf-8-sig')
        
        output_file = self._setup_output_file(len(df))
        
        # Определяем, сколько строк уже обработано (для безопасного возобновления)
        start_index = 0
        if os.path.exists(output_file):
            existing_data = pd.read_csv(output_file, sep=';', encoding='utf-8-sig')
            start_index = len(existing_data)
            print(f"Найдено {start_index} обработанных строк. Продолжаем...")

        print(f"Осталось обработать: {len(df) - start_index} записей.")

        # Цикл начинается ровно с того места, где скрипт остановился в прошлый раз
        for index, row in df.iloc[start_index:].iterrows():
            # Адаптируйте названия колонок под вашу исходную таблицу (raw_database)
            doc_id = str(row.get("ID", f"Документ_{index}"))
            text_input = str(row.get("Текст_документа", ""))
            
            # Пропуск пустых строк
            if len(text_input) < 10 or text_input.lower() in ['nan', 'none', '']:
                self._save_row(output_file, [doc_id, "Нет текста", "Нет", "Нет"])
                continue

            self._process_single_record(output_file, doc_id, text_input)

        print(f"\nВыгрузка завершена! Результаты лежат в: {output_file}")

    def _process_single_record(self, output_file, doc_id, text_input):
        """Обработка одной записи с механизмом повторных попыток (Retry)."""
        success = False
        retry_count = 0
        
        while not success:
            try:
                # Отправляем текст в оптимизированную нейросеть
                result = self.predictor(text_input=text_input)
                
                # Сразу сохраняем результат на диск
                self._save_row(output_file, [
                    doc_id, 
                    text_input, 
                    result.feature_1, 
                    result.feature_2
                ])
                
                print(f"Обработано: {doc_id}")
                success = True
                # Выдерживаем паузу из конфигурации
                time.sleep(config.execution["delay_seconds"]) 
                
            except Exception as e:
                retry_count += 1
                if retry_count >= config.execution["max_retries"]:
                    print(f"Критическая ошибка после {retry_count} попыток: {e}")
                    raise
                    
                print(f"Ошибка сервера. Ждем 10 секунд... Детали: {e}")
                time.sleep(10)

    def _save_row(self, output_file, data_list):
        """Вспомогательный метод для безопасной дозаписи строки в CSV."""
        with open(output_file, mode='a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(data_list)

# Блок для независимого тестирования модуля напрямую
if __name__ == "__main__":
    extractor = DataExtractor()
    extractor.run_extraction()