# ==========================================
# МОДУЛЬ ВЫБОРКИ (sampler.py)
# Случайный отбор статей для эталонов с удалением из основной базы.
# ==========================================

import pandas as pd
import config
import os

def create_samples():
    print("=== ПОДГОТОВКА ЭТАЛОНОВ И ЭКЗАМЕНА ===")
    
    # 1. Загрузка сырой базы
    # Используем utf-8-sig для защиты от проблем с кодировкой
    try:
        df = pd.read_csv(config.file_paths["raw_database"], sep=';', encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Ошибка: Исходный файл не найден по пути {config.file_paths['raw_database']}")
        return

    print(f"Исходная база загружена. Всего статей: {len(df)}")

    # 2. Случайная выборка 60 статей (random_state гарантирует случайность, но фиксирует ее при перезапусках)
    sampled_df = df.sample(n=60, random_state=42)
    
    # Разделяем пополам: 30 на учебу, 30 на экзамен
    golden_df = sampled_df.iloc[:30].copy()
    exam_df = sampled_df.iloc[30:].copy()

    # 3. Удаляем эти 60 статей из основной базы (физическая изоляция)
    baza_clean = df.drop(sampled_df.index)
    
    # Сохраняем очищенную базу (она заменит старую сырую)
    baza_clean.to_csv(config.file_paths["raw_database"], sep=';', encoding='utf-8-sig', index=False)
    print(f"Эталонные статьи физически удалены из основной базы. Осталось для выгрузки: {len(baza_clean)}")

    # 4. Вывод данных для ручной разметки в Excel
    # В Python индекс начинается с 0, а в Excel строка 1 — это заголовки. 
    # Поэтому точная строка в Excel = индекс Python + 2.
    
    print("\n" + "="*50)
    print(" 📚 ВАШИ 30 СТАТЕЙ ДЛЯ ОБУЧЕНИЯ (GOLDEN) ")
    print("="*50)
    for idx, row in golden_df.iterrows():
        excel_row = idx + 2
        title = str(row.get("Title", row.get("Название статьи", "Без названия")))
        print(f"Строка Excel №{excel_row} | {title[:80]}...")

    print("\n" + "="*50)
    print(" 🎓 ВАШИ 30 СТАТЕЙ ДЛЯ ЭКЗАМЕНА (EXAM) ")
    print("="*50)
    for idx, row in exam_df.iterrows():
        excel_row = idx + 2
        title = str(row.get("Title", row.get("Название статьи", "Без названия")))
        print(f"Строка Excel №{excel_row} | {title[:80]}...")

    # 5. Сохранение заготовок файлов для вас
    # Оставляем только нужные колонки, пустые места вы заполните руками (по 1-2 слова)
    columns_to_keep = ["Title", "Abstract"] # Адаптируйте под ваши заголовки
    
    # Добавляем пустые колонки для вашей разметки
    for col in ["Метод синтеза", "Область применения", "Роль диоксида", "Легирующая добавка"]:
        golden_df[col] = ""
        exam_df[col] = ""

    # Сохраняем файлы
    golden_df.to_csv(config.file_paths["golden_dataset"], sep=';', encoding='utf-8-sig', index=False)
    exam_df.to_csv(config.file_paths["exam_dataset"], sep=';', encoding='utf-8-sig', index=False)
    
    print(f"\n✅ Пустые шаблоны golden.csv и exam.csv успешно созданы и ждут вашей ручной разметки!")

if __name__ == "__main__":
    create_samples()