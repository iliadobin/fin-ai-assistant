"""
Модуль для загрузки данных из CSV файлов.
"""

import pandas as pd


def load_data(train_file="./train_data.csv", questions_file="./questions.csv"):
    """
    Загружает данные из CSV файлов.
    
    Args:
        train_file (str): Путь к файлу с базой знаний (статьи)
        questions_file (str): Путь к файлу с вопросами
    
    Returns:
        tuple: (train_data DataFrame, questions DataFrame)
    """
    print("Загрузка данных...")
    
    # Загружаем базу знаний (финансовые статьи)
    train_data = pd.read_csv(train_file)
    print(f"Загружено статей: {len(train_data)}")
    print(f"Колонки в train_data: {train_data.columns.tolist()}")
    
    # Загружаем вопросы
    questions = pd.read_csv(questions_file)
    print(f"Загружено вопросов: {len(questions)}")
    print(f"Колонки в questions: {questions.columns.tolist()}")
    
    return train_data, questions

