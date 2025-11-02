"""
Модуль для поиска релевантных статей по векторным представлениям (embeddings).
Реализует семантический поиск через cosine similarity.
"""

import numpy as np
from embeddings import get_embedding


def cosine_similarity(vec1, vec2):
    """
    Вычисляет косинусное сходство между двумя векторами.
    
    Args:
        vec1 (np.array): Первый вектор
        vec2 (np.array): Второй вектор
        
    Returns:
        float: Косинусное сходство (от -1 до 1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def find_most_relevant_article(question, train_data, article_embeddings, api_key):
    """
    Находит наиболее релевантную статью для заданного вопроса.
    
    Args:
        question (str): Вопрос пользователя
        train_data (DataFrame): Данные со статьями
        article_embeddings (np.array): Матрица embeddings статей
        api_key (str): API ключ для embedder модели
        
    Returns:
        dict: Словарь с информацией о найденной статье
              {'id': id, 'text': text, 'annotation': annotation, 'similarity': score}
    """
    # Получаем embedding для вопроса
    question_embedding = np.array(get_embedding(question, api_key))
    
    # Вычисляем косинусное сходство со всеми статьями
    similarities = []
    for article_emb in article_embeddings:
        similarity = cosine_similarity(question_embedding, article_emb)
        similarities.append(similarity)
    
    # Находим индекс статьи с максимальным сходством
    similarities = np.array(similarities)
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    
    # Получаем информацию о найденной статье
    best_article = train_data.iloc[best_idx]
    
    return {
        'id': best_article['id'],
        'text': best_article['text'],
        'annotation': best_article['annotation'],
        'tags': best_article['tags'],
        'similarity': best_similarity
    }

