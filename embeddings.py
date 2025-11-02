"""
Модуль для работы с embeddings через EMBEDDER API.
Включает функции создания, сохранения и загрузки векторных представлений текстов.
"""

import numpy as np
from openai import OpenAI
from tqdm import tqdm
import pickle
import os


def get_embedding(text, api_key):
    """
    Получает embedding для одного текста через EMBEDDER API.
    
    Args:
        text (str): Текст для векторизации
        api_key (str): API ключ для embedder модели
        
    Returns:
        list: Вектор embedding
    """
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=api_key,
    )
    
    # Обрабатываем случаи, когда текст пустой или NaN
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # Заменяем переносы строк на пробелы для корректной обработки
    text = text.replace("\n", " ").strip()
    
    # Если текст пустой, используем placeholder
    if not text:
        text = "Empty text"
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    
    return response.data[0].embedding


def create_embeddings_batch(texts, api_key, batch_size=10):
    """
    Создает embeddings для списка текстов с батчингом и progress bar.
    
    Args:
        texts (list): Список текстов для векторизации
        api_key (str): API ключ для embedder модели
        batch_size (int): Размер батча для обработки
        
    Returns:
        np.array: Матрица embeddings (n_texts, embedding_dim)
    """
    embeddings = []
    
    print(f"Создание embeddings для {len(texts)} текстов...")
    
    # Обрабатываем тексты батчами с progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Создание embeddings"):
        batch = texts[i:i + batch_size]
        
        for text in batch:
            embedding = get_embedding(text, api_key)
            embeddings.append(embedding)
    
    return np.array(embeddings)


def save_embeddings(embeddings, texts_ids, filepath):
    """
    Сохраняет embeddings в файл.
    
    Args:
        embeddings (np.array): Матрица embeddings
        texts_ids (list): Список ID текстов (для проверки соответствия)
        filepath (str): Путь для сохранения файла
    """
    data = {
        'embeddings': embeddings,
        'ids': texts_ids
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Embeddings сохранены в {filepath}")


def load_embeddings(filepath):
    """
    Загружает embeddings из файла.
    
    Args:
        filepath (str): Путь к файлу с embeddings
        
    Returns:
        tuple: (embeddings np.array, ids list) или (None, None) если файл не найден
    """
    if not os.path.exists(filepath):
        return None, None
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Embeddings загружены из {filepath}")
    return data['embeddings'], data['ids']


def load_or_create_embeddings(train_data, embeddings_file, api_key, use_annotation=True):
    """
    Загружает существующие embeddings или создает новые.
    
    Args:
        train_data (DataFrame): Данные со статьями
        embeddings_file (str): Путь к файлу с embeddings
        api_key (str): API ключ для embedder модели
        use_annotation (bool): Использовать annotation вместо text (для длинных текстов)
        
    Returns:
        np.array: Матрица embeddings
    """
    # Пытаемся загрузить существующие embeddings
    embeddings, saved_ids = load_embeddings(embeddings_file)
    
    # Проверяем, совпадают ли ID статей
    current_ids = train_data['id'].tolist()
    
    if embeddings is not None and saved_ids == current_ids:
        print("Используем существующие embeddings")
        return embeddings
    
    # Создаем новые embeddings
    print("Создание новых embeddings...")
    
    # Используем annotation для embeddings (короче и содержит суть статьи)
    # Полный text будет использоваться при генерации ответов
    if use_annotation:
        texts = train_data['annotation'].tolist()
        print("(используем annotation для embeddings - короче и быстрее)")
    else:
        texts = train_data['text'].tolist()
    
    embeddings = create_embeddings_batch(texts, api_key)
    
    # Сохраняем для будущего использования
    save_embeddings(embeddings, current_ids, embeddings_file)
    
    return embeddings

