import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import pickle

# Подключаем все переменные из окружения
load_dotenv()

# Подключаем ключи для API
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

# Константы
EMBEDDINGS_FILE = "embeddings.pkl"
TRAIN_DATA_FILE = "./train_data.csv"
QUESTIONS_FILE = "./questions.csv"
OUTPUT_FILE = "submission.csv"


def load_data():
    """
    Загружает данные из CSV файлов.
    
    Returns:
        tuple: (train_data DataFrame, questions DataFrame)
    """
    print("Загрузка данных...")
    
    # Загружаем базу знаний (финансовые статьи)
    train_data = pd.read_csv(TRAIN_DATA_FILE)
    print(f"Загружено статей: {len(train_data)}")
    print(f"Колонки в train_data: {train_data.columns.tolist()}")
    
    # Загружаем вопросы
    questions = pd.read_csv(QUESTIONS_FILE)
    print(f"Загружено вопросов: {len(questions)}")
    print(f"Колонки в questions: {questions.columns.tolist()}")
    
    return train_data, questions


if __name__ == "__main__":
    # Тестовая загрузка данных
    train_data, questions = load_data()
    
    # Выводим примеры для проверки
    print("\n=== Пример статьи ===")
    print(f"ID: {train_data.iloc[0]['id']}")
    print(f"Аннотация: {train_data.iloc[0]['annotation'][:200]}...")
    print(f"Длина текста: {len(train_data.iloc[0]['text'])} символов")
    
    print("\n=== Пример вопроса ===")
    print(f"ID: {questions.iloc[0]['ID вопроса']}")
    print(f"Вопрос: {questions.iloc[0]['Вопрос']}")

