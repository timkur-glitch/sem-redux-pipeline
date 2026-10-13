# ==========================================
# МОДУЛЬ ОБУЧЕНИЯ (trainer.py)
# Отвечает за алгоритмическую оптимизацию промптов (DSPy).
# ==========================================

import pandas as pd
import dspy
from dspy.teleprompt import BootstrapFewShot

# Импортируем наш единый файл конфигурации
import config 

# 1. СХЕМА ИЗВЛЕЧЕНИЯ ДАННЫХ (SIGNATURE)
# Шаблон, определяющий, что именно модель должна найти в тексте.
class ExtractionSignature(dspy.Signature):
    """Прочитай исходный текст и извлеки целевые характеристики в 1-2 слова."""
    
    # Входные данные (условие задачи)
    text_input = dspy.InputField(desc="Исходный текст для анализа")
    
    # Ожидаемые выходные данные (пользователь может добавлять свои категории)
    feature_1 = dspy.OutputField(desc="Первая характеристика (1-2 слова)")
    feature_2 = dspy.OutputField(desc="Вторая характеристика (1-2 слова)")


# 2. ИНКАПСУЛЯЦИЯ ЛОГИКИ В КЛАСС (ООП)
class ModelTrainer:
    def __init__(self):
        # Инициализируем языковую модель, подтягивая ключ и название из config.py
        self.language_model = dspy.LM(config.model_name, api_key=config.api_key)
        dspy.settings.configure(lm=self.language_model)
        self.predictor = dspy.Predict(ExtractionSignature)

    @staticmethod
    def exact_match_metric(example, prediction, trace=None):
        """Метрика (судья), строго проверяющая совпадение ответа ИИ с эталоном."""
        score = 0
        if example.feature_1.lower() in prediction.feature_1.lower(): score += 1
        if example.feature_2.lower() in prediction.feature_2.lower(): score += 1
        
        # Делим на количество признаков для получения балла от 0 до 1.0
        return score / 2.0

    def load_training_data(self):
        """Загружает эталонный файл и переводит его в формат DSPy."""
        print("Загрузка эталонного набора данных...")
        
        # Читаем путь к файлу golden.csv из конфигурации
        df = pd.read_csv(config.file_paths["golden_dataset"], sep=';', encoding='utf-8-sig')
        
        training_set = []
        for _, row in df.iterrows():
            # Названия колонок (row["..."]) пользователь должен адаптировать под свою таблицу
            example = dspy.Example(
                text_input=str(row["Текст_документа"]),
                feature_1=str(row["Целевой_признак_1"]).strip(),
                feature_2=str(row["Целевой_признак_2"]).strip()
            ).with_inputs("text_input")
            
            training_set.append(example)
            
        return training_set

    def run_optimization(self):
        """Запускает алгоритм BootstrapFewShot и сохраняет готовую модель."""
        training_set = self.load_training_data()
        
        print("\nНастройка компилятора DSPy...")
        # Подтягиваем гиперпараметры обучения из config.py
        optimizer = BootstrapFewShot(
            metric=self.exact_match_metric, 
            max_bootstrapped_demos=config.dspy_settings["bootstrapped_demos"], 
            max_labeled_demos=config.dspy_settings["labeled_demos"]
        )
        
        print("Запуск алгоритмической оптимизации (обучения)...")
        compiled_model = optimizer.compile(self.predictor, trainset=training_set)
        
        # Сохраняем "поумневшую" модель по пути, указанному в config.py
        save_path = config.file_paths["optimized_model"]
        compiled_model.save(save_path)
        print(f"\nМодель успешно обучена и сохранена в файл: {save_path}")

# Блок для независимого тестирования модуля
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_optimization()