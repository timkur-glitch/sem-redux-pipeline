# ==========================================
# МОДУЛЬ ОЦЕНКИ (evaluator.py)
# Независимая проверка качества на экзаменационной выборке.
# ==========================================

import os
import pandas as pd
import dspy
from sklearn.metrics import cohen_kappa_score

# Импортируем наш файл конфигурации
import config

# 1. СХЕМА ИЗВЛЕЧЕНИЯ (Должна совпадать с trainer.py и extractor.py)
class ExtractionSignature(dspy.Signature):
    """Прочитай исходный текст и извлеки целевые характеристики в 1-2 слова."""
    text_input = dspy.InputField(desc="Исходный текст для анализа")
    feature_1 = dspy.OutputField(desc="Первая характеристика (1-2 слова)")
    feature_2 = dspy.OutputField(desc="Вторая характеристика (1-2 слова)")


class ModelEvaluator:
    def __init__(self):
        """Инициализация модели и загрузка оптимизированных параметров."""
        self.language_model = dspy.LM(config.model_name, api_key=config.api_key)
        dspy.settings.configure(lm=self.language_model)
        
        self.predictor = dspy.Predict(ExtractionSignature)
        
        model_path = config.file_paths["optimized_model"]
        if os.path.exists(model_path):
            self.predictor.load(model_path)
            print(f"Обученная модель загружена для экзамена из: {model_path}")
        else:
            raise FileNotFoundError("Обученная модель не найдена. Запустите trainer.py")

    def run_evaluation(self):
        """Прогоняет экзаменационную выборку и рассчитывает метрики."""
        print("Загрузка экзаменационной выборки (exam dataset)...")
        # Читаем файл с экзаменом (он содержит и тексты, и правильные ответы эксперта)
        df_exam = pd.read_csv(config.file_paths["exam_dataset"], sep=';', encoding='utf-8-sig')
        
        expert_answers_f1 = []
        ai_answers_f1 = []
        expert_answers_f2 = []
        ai_answers_f2 = []

        print(f"Начинаем независимое тестирование ({len(df_exam)} записей)...")
        
        for index, row in df_exam.iterrows():
            text_input = str(row.get("Текст_документа", ""))
            
            # Скрытые ответы эксперта (для проверки)
            expert_f1 = str(row.get("Целевой_признак_1", "")).strip().lower()
            expert_f2 = str(row.get("Целевой_признак_2", "")).strip().lower()
            
            # Просим ИИ дать свой ответ только на основе текста
            try:
                result = self.predictor(text_input=text_input)
                ai_f1 = result.feature_1.strip().lower()
                ai_f2 = result.feature_2.strip().lower()
            except Exception as e:
                print(f"Ошибка при обработке строки {index}: {e}")
                ai_f1, ai_f2 = "ошибка", "ошибка"

            # Сохраняем для финального подсчета
            expert_answers_f1.append(expert_f1)
            ai_answers_f1.append(ai_f1)
            expert_answers_f2.append(expert_f2)
            ai_answers_f2.append(ai_f2)

        self._calculate_metrics(expert_answers_f1, ai_answers_f1, "Признак 1")
        self._calculate_metrics(expert_answers_f2, ai_answers_f2, "Признак 2")

    def _calculate_metrics(self, expert_list, ai_list, feature_name):
        """Вспомогательный метод для расчета Каппы Коэна и точности."""
        # Расчет совпадений
        exact_matches = sum(1 for e, a in zip(expert_list, ai_list) if e == a)
        accuracy = exact_matches / len(expert_list)
        
        # Расчет Каппы Коэна через sklearn
        kappa = cohen_kappa_score(expert_list, ai_list)
        
        print(f"\n=== Результаты для: {feature_name} ===")
        print(f"Точное совпадение (Accuracy): {accuracy:.2f}")
        print(f"Межэкспертная надежность (Cohen's Kappa): {kappa:.3f}")

# Блок для независимого тестирования модуля
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()