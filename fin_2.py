import requests
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime, timedelta


# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = 'H5WE0MV3NWQT0ZM4'

# Load pre-trained FinBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)


# Function to get continuous sentiment score using FinBERT with softmax
def get_finbert_sentiment_score(text):
    """
    This function calculates sentiment probabilities using softmax on FinBERT output logits.

    Parameters:
    - text: str, the text content for which sentiment needs to be calculated.

    Returns:
    - dict: A dictionary containing sentiment probabilities for negative, neutral, and positive.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    # Extract logits and apply softmax to convert to probabilities
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1).detach().numpy()[0]

    # Extract probabilities for each class
    negative_prob, neutral_prob, positive_prob = probabilities

    # Return sentiment probabilities and a simplified sentiment score (positive - negative)
    return {
        'negative_prob': negative_prob,
        'neutral_prob': neutral_prob,
        'positive_prob': positive_prob,
        'sentiment_score': positive_prob - negative_prob  # Sentiment score: positive - negative
    }


# Function to fetch news articles using Alpha Vantage's NEWS_SENTIMENT API
def fetch_news_from_alpha_vantage(ticker, time_from, time_to, max_articles=100):
    """
    Fetches news articles using Alpha Vantage's NEWS_SENTIMENT API.

    Parameters:
    - ticker: str, the stock ticker symbol (e.g., 'AAPL' for Apple).
    - time_from: str, the start date in the format YYYYMMDDTHHMM.
    - time_to: str, the end date in the format YYYYMMDDTHHMM.
    - max_articles: int, the maximum number of articles to retrieve.

    Returns:
    - articles: List of dictionaries containing the news article summaries.
    """
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={time_from}&time_to={time_to}&limit={max_articles}&apikey={ALPHA_VANTAGE_API_KEY}"

    response = requests.get(url)

    if response.status_code != 200:
        print("Error fetching news from Alpha Vantage:", response.status_code)
        return []

    data = response.json()

    if 'feed' not in data:
        print("No news data found for the given ticker.")
        return []

    articles = data['feed']
    return articles


# Function to calculate average sentiment score for fetched news articles
def calculate_average_sentiment(articles):
    """
    Calculates the average sentiment score for a list of articles using FinBERT.

    Parameters:
    - articles: List of news article summaries.

    Returns:
    - average_sentiment: float, the average sentiment score for the articles.
    """
    sentiment_scores = []

    for article in articles:
        sentiment_result = get_finbert_sentiment_score(article['summary'])
        sentiment_scores.append(sentiment_result['sentiment_score'])

    # Calculate the average sentiment score
    if sentiment_scores:
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    else:
        average_sentiment = 0

    return average_sentiment


# Define date range for the last four months
end_date = datetime.now()
start_date = end_date - timedelta(days=120)

# Format the dates for Alpha Vantage API (format: YYYYMMDDTHHMM)
time_from = start_date.strftime('%Y%m%dT%H%M')
time_to = end_date.strftime('%Y%m%dT%H%M')

# Example: Fetch news articles about 'Apple' from Alpha Vantage
ticker = 'AAPL'
articles = fetch_news_from_alpha_vantage(ticker, time_from, time_to)

# Check if articles were retrieved
if articles:
    print(f"Fetched {len(articles)} news articles for {ticker}")

    # Calculate average sentiment score for the articles
    average_sentiment = calculate_average_sentiment(articles)
    print(f"Average Sentiment Score for {ticker} news articles: {average_sentiment:.4f}")

    # Optional: Save articles and sentiment scores to CSV for further analysis
    df = pd.DataFrame(articles)
    df.to_csv(f'{ticker}_news_sentiment.csv', index=False)
else:
    print(f"No news articles found for {ticker}")
