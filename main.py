from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Импортируем модули проекта
from data_loader import load_data
from embeddings import load_or_create_embeddings
from search import find_most_relevant_article

# Подключаем все переменные из окружения
load_dotenv()

# Подключаем ключи для API
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

# Проверяем загрузку ключей
if not EMBEDDER_API_KEY:
    raise ValueError("EMBEDDER_API_KEY не найден в .env файле! Добавьте строку: EMBEDDER_API_KEY=sk-0y14H33guZQVnRZFVE6BzQ")
if not LLM_API_KEY:
    raise ValueError("LLM_API_KEY не найден в .env файле!")

# Константы
EMBEDDINGS_FILE = "embeddings.pkl"
TRAIN_DATA_FILE = "./train_data.csv"
QUESTIONS_FILE = "./questions.csv"
OUTPUT_FILE = "submission.csv"


def answer_generation(question, article_text, api_key):
    """
    Генерирует ответ на вопрос на основе релевантной статьи (RAG подход).
    
    Args:
        question (str): Вопрос пользователя
        article_text (str): Текст релевантной статьи для контекста
        api_key (str): API ключ для LLM модели
        
    Returns:
        str: Сгенерированный ответ
    """
    # Подключаемся к LLM модели
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=api_key,
    )
    
    # Формируем промпт с контекстом статьи
    prompt = f"""Ты - финансовый AI-ассистент банка. Твоя задача - давать точные и грамотные ответы на вопросы клиентов о финансовых инструментах и услугах.

Используй ТОЛЬКО информацию из предоставленной статьи для ответа. Не добавляй информацию, которой нет в статье.

СТАТЬЯ:
{article_text}

ВОПРОС КЛИЕНТА:
{question}

Дай четкий, профессиональный и понятный ответ на русском языке. Ответ должен быть информативным, но кратким (2-4 предложения)."""

    # Формируем запрос к LLM
    response = client.chat.completions.create(
        model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    
    # Возвращаем сгенерированный ответ
    return response.choices[0].message.content


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ФИНАНСОВЫЙ AI-АССИСТЕНТ - RAG система")
    print("="*80 + "\n")
    
    # Шаг 1: Загружаем данные
    print("📂 Шаг 1/4: Загрузка данных...")
    train_data, questions = load_data(TRAIN_DATA_FILE, QUESTIONS_FILE)
    
    # Шаг 2: Создаем/загружаем embeddings для всех статей
    print("\n🔢 Шаг 2/4: Подготовка embeddings...")
    article_embeddings = load_or_create_embeddings(train_data, EMBEDDINGS_FILE, EMBEDDER_API_KEY)
    
    # Шаг 3: Обрабатываем все вопросы
    print(f"\n🤖 Шаг 3/4: Генерация ответов на {len(questions)} вопросов...")
    print("(это займет ~15-20 минут)\n")
    
    answer_list = []
    
    for idx, row in tqdm(questions.iterrows(), total=len(questions), desc="Обработка вопросов"):
        current_question = row['Вопрос']
        
        # Находим наиболее релевантную статью
        relevant_article = find_most_relevant_article(
            current_question,
            train_data,
            article_embeddings,
            EMBEDDER_API_KEY
        )
        
        # Генерируем ответ на основе найденной статьи
        answer = answer_generation(
            current_question,
            relevant_article['text'],
            LLM_API_KEY
        )
        
        answer_list.append(answer)
    
    # Шаг 4: Сохраняем результаты
    print("\n💾 Шаг 4/4: Сохранение результатов...")
    questions['Ответы на вопрос'] = answer_list
    questions.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Готово! Результаты сохранены в {OUTPUT_FILE}")
    print("="*80)

