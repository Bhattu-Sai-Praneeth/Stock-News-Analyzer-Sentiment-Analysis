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
from urllib.parse import quote

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')

# -----------------------------
# 1) Configuration
# -----------------------------
NEWS_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_finbert():
    return pipeline("text-classification", model="ProsusAI/finbert")

def get_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session_news = get_session()

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
]

# -----------------------------
# 2) News Fetching (Fixed)
# -----------------------------

def fetch_news_newsdata(company):
    """Fetch news from NewsData.io with improved error handling"""
    url = "https://newsdata.io/api/1/news"
    
    params = {
        "apikey": NEWS_API_KEY,
        "q": quote(company),  # Proper encoding
        "language": "en",
        "page": 1
    }
    
    try:
        response = session_news.get(url, params=params, timeout=15)
        st.write(f"Request URL: {response.url}")  # Debugging info
        response.raise_for_status()
        
        data = response.json()
        if "results" not in data:
            st.error(f"Unexpected API response: {data}")
            return []

        return [(art.get("title", "No Title"), art.get("description", "No summary available"), art.get("link", ""))
                for art in data["results"][:5]]
    
    except requests.exceptions.RequestException as e:
        st.error(f"NewsData.io API Error: {e}")
        return []

def scrape_google_news(company):
    """Fetch news from Google News RSS"""
    url = f"https://news.google.com/rss/search?q={quote(company)}&hl=en-IN&gl=IN&ceid=IN:en"
    
    try:
        response = session_news.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'xml')
        return [(item.title.text, item.description.text if item.description else "No summary available", item.link.text)
                for item in soup.find_all('item')[:5]]
    except Exception as e:
        st.error(f"Google News Error: {e}")
        return []

# -----------------------------
# 3) Article Parsing
# -----------------------------

def parse_article_content(url):
    """Extract content from news articles"""
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        response = session_news.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs])[:5000]  # Limit to 5000 characters
    except Exception:
        return ""

# -----------------------------
# 4) Text Summarization
# -----------------------------

def generate_summary(text):
    """Summarize extracted article text"""
    if not text or len(text) < 100:
        return "No summary available"
    
    try:
        summarizer = load_summarizer()
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception:
        return "Summary generation failed"

# -----------------------------
# 5) Sentiment Analysis
# -----------------------------

def analyze_sentiment(text, method):
    """Analyze sentiment using VADER or FinBERT"""
    if not text:
        return "Neutral"
    
    clean_text = ' '.join(text.split()[:512])  # Truncate to 512 words
    
    try:
        if method == "VADER":
            scores = load_vader().polarity_scores(clean_text)
            return "Positive" if scores['compound'] >= 0.05 else "Negative" if scores['compound'] <= -0.05 else "Neutral"
        elif method == "FinBERT":
            result = load_finbert()(clean_text, truncation=True)[0]
            return result['label'].capitalize()
    except Exception:
        return "Neutral"

# -----------------------------
# 6) Fetch & Analyze News
# -----------------------------

def fetch_and_analyze_news(company, method="VADER", use_newsdata=False):
    """Fetch, summarize, and analyze sentiment of news"""
    sources = [fetch_news_newsdata(company)] if use_newsdata else [scrape_google_news(company)]
    
    articles = [article for source in sources for article in source]
    
    if not articles:
        return "Neutral", []

    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    analyzed_news = []

    for title, summary, link in articles:
        summary = summary or generate_summary(parse_article_content(link))
        sentiment = analyze_sentiment(f"{title}. {summary}", method)
        
        sentiment_counts[sentiment] += 1
        analyzed_news.append((title, summary, sentiment, link))

    total = sum(sentiment_counts.values())
    overall = ("Positive" if sentiment_counts["Positive"] / total > 0.4 else
               "Negative" if sentiment_counts["Negative"] / total > 0.4 else "Neutral") if total > 0 else "Neutral"

    return overall, analyzed_news

# -----------------------------
# 7) Streamlit UI
# -----------------------------

st.title("📈 Stock News Sentiment Analyzer")

company = st.text_input("Enter Company Name", "Reliance Industries")
method = st.selectbox("Sentiment Analysis Method", ["VADER", "FinBERT"])
use_newsdata = st.checkbox("Use NewsData.io API (requires valid API key)")

if st.button("Analyze News Sentiment"):
    with st.spinner("Gathering and analyzing news..."):
        start_time = time.time()
        overall, articles = fetch_and_analyze_news(company, method, use_newsdata)
    
    st.subheader(f"Overall Sentiment: **{overall}**")
    
    # Display sentiment distribution
    counts = pd.DataFrame({"Sentiment": ["Positive", "Negative", "Neutral"],
                           "Count": [sum(1 for a in articles if a[2] == "Positive"),
                                     sum(1 for a in articles if a[2] == "Negative"),
                                     sum(1 for a in articles if a[2] == "Neutral")]})
    st.bar_chart(counts.set_index("Sentiment"))
    
    # Display news articles
    st.subheader("News Analysis")
    for idx, (title, summary, sentiment, link) in enumerate(articles, 1):
        with st.expander(f"{idx}. {sentiment} - {title[:70]}..."):
            st.markdown(f"**Summary:** {summary}")
            st.markdown(f"**Sentiment:** {sentiment}")
            if link:
                st.markdown(f"[Read full article ↗️]({link})")

st.info("This tool uses web scraping; results depend on news availability and sources.")
