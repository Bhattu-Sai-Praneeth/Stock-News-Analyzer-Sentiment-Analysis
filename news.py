import streamlit as st
import requests
import random
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Ensure NLTK downloads
nltk.download('vader_lexicon')

# API Key for NewsData.io
NEWS_API_KEY = "pub_726340e45067fcad1d9a6d2fef24ba983aab3"

# Setup session for retries
def get_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session_news = get_session()

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
]

# --- Load Sentiment Models ---
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_finbert():
    return pipeline("text-classification", model="ProsusAI/finbert")

# --- Fetch News from NewsData.io ---
def fetch_news_newsdata(company):
    url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&q={company}&language=en&country=in&page=1"
    try:
        response = session_news.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return [(article["title"], article["description"], article["link"]) for article in data.get("results", [])[:5]]
    except Exception:
        return []

# --- Scrape MoneyControl News ---
def scrape_moneycontrol_news(company):
    search_url = f"https://www.moneycontrol.com/news/tags/{company.replace(' ', '-').lower()}.html"
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        response = session_news.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('li', class_='clearfix')[:5]
        return [(article.find('h2').text.strip(), "No summary available", article.find('a')['href']) for article in articles if article.find('h2')]
    except Exception:
        return []

# --- Scrape Bing News ---
def scrape_bing_news(company):
    search_url = f"https://www.bing.com/news/search?q={company.replace(' ', '+')}"
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        response = session_news.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all("div", class_="news-card")[:5]
        return [(a.find("a").get_text(strip=True), "No summary available", a.find("a")["href"]) for a in articles if a.find("a")]
    except Exception:
        return []

# --- Sentiment Analysis ---
def analyze_sentiment(text, method):
    try:
        if method == "VADER":
            sia = load_vader()
            compound = sia.polarity_scores(text)['compound']
            return "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
        elif method == "FinBERT":
            finbert = load_finbert()
            result = finbert(text[:512], truncation=True)[0]
            return result['label'].capitalize()
    except Exception:
        return "Error"

# --- News Aggregation & Sentiment Calculation ---
def fetch_and_analyze_news(company, method="VADER", use_newsdata=False, aggregate_sources=False):
    news_sources = []
    if aggregate_sources:
        news_sources = fetch_news_newsdata(company) + scrape_moneycontrol_news(company) + scrape_bing_news(company)
    elif use_newsdata:
        news_sources = fetch_news_newsdata(company)
    else:
        news_sources = scrape_moneycontrol_news(company) + scrape_bing_news(company)

    if not news_sources:
        return "Neutral", []

    pos, neg, neu = 0, 0, 0
    analyzed_news = []

    for headline, summary, link in news_sources:
        sentiment = analyze_sentiment(headline, method).capitalize()  # Ensure uniform sentiment labels
        analyzed_news.append((headline, summary, sentiment, link))

        if sentiment not in ["Positive", "Negative", "Neutral"]:
            sentiment = "Neutral"  # Handle unexpected sentiment values

        if sentiment == "Positive":
            pos += 1
        elif sentiment == "Negative":
            neg += 1
        else:
            neu += 1

    total = pos + neg + neu
    if total == 0:
        return "Neutral", analyzed_news
    if (pos / total) > 0.4:
        return "Positive", analyzed_news
    elif (neg / total) > 0.4:
        return "Negative", analyzed_news
    else:
        return "Neutral", analyzed_news

# --- Streamlit UI ---
st.title("📊 Stock Market News Sentiment Analysis")

company = st.text_input("Enter Company Name", "HDFC Bank")

col1, col2 = st.columns(2)
with col1:
    sentiment_method = st.radio("Select Sentiment Analysis Model", ("VADER", "FinBERT"), index=0)
with col2:
    use_newsdata = st.checkbox("Use NewsData.io for news", value=False)
    aggregate_sources = st.checkbox("Aggregate Multiple Sources", value=False)

if st.button("Analyze News Sentiment"):
    with st.spinner(f"Fetching news for {company}..."):
        sentiment, news_list = fetch_and_analyze_news(company, sentiment_method, use_newsdata, aggregate_sources)

    st.subheader(f"Overall Sentiment for {company}: **{sentiment}**")

    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for _, _, sentiment, _ in news_list:
        sentiment_counts[sentiment] += 1

    st.bar_chart(pd.DataFrame(sentiment_counts, index=["Sentiment"]).T)

    for headline, summary, sentiment, link in news_list:
        with st.expander(f"📰 {headline}"):
            st.write(f"**Summary:** {summary}")
            st.write(f"**Sentiment:** **{sentiment}**")
            st.markdown(f"[🔗 Read Article]({link})")
