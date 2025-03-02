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
# Replace with your valid NewsData.io API key
NEWS_API_KEY "pub_726340e45067fcad1d9a6d2fef24ba983aab3"

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
# 2) News Fetching (Improved)
# -----------------------------

def fetch_news_newsdata(company):
    """Improved with proper URL encoding and error handling"""
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": NEWS_API_KEY,
        "q": company,
        "language": "en",
        "page": 1
    }
    
    try:
        response = session_news.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for art in data.get("results", [])[:5]:
            title = art.get("title", "No Title")
            desc = art.get("description") or "No summary available"
            link = art.get("link", "")
            results.append((title, desc, link))
        return results
    except Exception as e:
        st.error(f"NewsData.io API Error: {str(e)}")
        return []

def scrape_moneycontrol_news(company):
    """Updated CSS selectors for MoneyControl"""
    formatted_company = company.replace(' ', '-').lower()
    search_url = f"https://www.moneycontrol.com/news/tags/{formatted_company}.html"
    
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        response = session_news.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('li', class_='clearfix')[:5]
        
        results = []
        for art in articles:
            title_tag = art.find('h2')
            if not title_tag:
                continue
                
            title = title_tag.text.strip()
            link_tag = art.find('a', href=True)
            link = link_tag['href'] if link_tag else ""
            
            # Extract summary from sibling div
            summary_tag = art.find('p', class_='news_desc')
            summary = summary_tag.text.strip() if summary_tag else None
            
            results.append((title, summary, link))
        return results
    except Exception as e:
        st.error(f"MoneyControl Error: {str(e)}")
        return []

def scrape_google_news(company):
    """Alternative news source using Google News"""
    formatted_query = quote(company)
    url = f"https://news.google.com/rss/search?q={formatted_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    try:
        response = session_news.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'xml')
        items = soup.find_all('item')[:5]
        
        results = []
        for item in items:
            title = item.title.text if item.title else "No Title"
            link = item.link.text if item.link else ""
            description = item.description.text if item.description else None
            results.append((title, description, link))
        return results
    except Exception as e:
        st.error(f"Google News Error: {str(e)}")
        return []

# -----------------------------
# 3) Enhanced Article Parsing
# -----------------------------

def parse_article_content(url):
    """Improved content extraction with better HTML parsing"""
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        response = session_news.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find article body using common semantic tags
        article_body = soup.find('article') or soup.find('div', class_='article-content')
        
        if not article_body:
            # Fallback to paragraph extraction
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
        else:
            text = ' '.join([p.get_text() for p in article_body.find_all('p')])
        
        return text.strip()[:5000]  # Limit to 5000 characters
    except Exception:
        return ""

# -----------------------------
# 4) Enhanced Summarization
# -----------------------------

def generate_summary(text):
    """Improved summarization with error handling"""
    if not text or len(text) < 100:
        return "No summary available"
    
    try:
        summarizer = load_summarizer()
        summary = summarizer(
            text,
            max_length=100,
            min_length=30,
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception:
        return "Summary generation failed"

# -----------------------------
# 5) Sentiment Analysis
# -----------------------------

def analyze_sentiment(text, method):
    """Enhanced with text preprocessing"""
    if not text:
        return "Neutral"
    
    # Preprocess text
    clean_text = ' '.join(text.split()[:512])  # Truncate to 512 words
    
    try:
        if method == "VADER":
            sia = load_vader()
            scores = sia.polarity_scores(clean_text)
            if scores['compound'] >= 0.05:
                return "Positive"
            elif scores['compound'] <= -0.05:
                return "Negative"
            return "Neutral"
        elif method == "FinBERT":
            finbert = load_finbert()
            result = finbert(clean_text, truncation=True)[0]
            return result['label'].capitalize()
    except Exception:
        return "Neutral"

# -----------------------------
# 6) Main Processing Function
# -----------------------------

def fetch_and_analyze_news(company, method="VADER", use_newsdata=False):
    """Improved news aggregation and processing"""
    sources = []
    
    if use_newsdata:
        sources.append(fetch_news_newsdata(company))
    else:
        sources.append(scrape_moneycontrol_news(company))
        sources.append(scrape_google_news(company))
    
    all_articles = [article for source in sources for article in source]
    
    if not all_articles:
        return "Neutral", []
    
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    analyzed_news = []
    
    for title, summary, link in all_articles:
        # Generate summary if missing
        if not summary:
            article_text = parse_article_content(link)
            summary = generate_summary(article_text) if article_text else "No summary available"
        
        # Analyze sentiment using both title and summary
        combined_text = f"{title}. {summary}"
        sentiment = analyze_sentiment(combined_text, method)
        
        sentiment_counts[sentiment] += 1
        analyzed_news.append((title, summary, sentiment, link))
    
    # Determine overall sentiment
    total = sum(sentiment_counts.values())
    if total == 0:
        return "Neutral", analyzed_news
    
    pos_ratio = sentiment_counts["Positive"] / total
    neg_ratio = sentiment_counts["Negative"] / total
    
    if pos_ratio > 0.4:
        overall = "Positive"
    elif neg_ratio > 0.4:
        overall = "Negative"
    else:
        overall = "Neutral"
    
    return overall, analyzed_news

# -----------------------------
# 7) Streamlit UI
# -----------------------------

st.title("📈 Advanced Stock News Sentiment Analyzer")

company = st.text_input("Enter Company Name", "Reliance Industries")
method = st.selectbox("Sentiment Analysis Method", ["VADER", "FinBERT"])
use_newsdata = st.checkbox("Use NewsData.io API (requires valid API key)")

if st.button("Analyze News Sentiment"):
    with st.spinner("Gathering and analyzing news..."):
        start_time = time.time()
        overall, articles = fetch_and_analyze_news(
            company,
            method=method,
            use_newsdata=use_newsdata
        )
    
    st.subheader(f"Overall Sentiment: **{overall}**")
    
    # Display sentiment distribution
    counts = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Count": [
            sum(1 for a in articles if a[2] == "Positive"),
            sum(1 for a in articles if a[2] == "Negative"),
            sum(1 for a in articles if a[2] == "Neutral")
        ]
    })
    st.bar_chart(counts.set_index("Sentiment"))
    
    # Display individual articles
    st.subheader("Detailed News Analysis")
    for idx, (title, summary, sentiment, link) in enumerate(articles, 1):
        with st.expander(f"{idx}. {sentiment} - {title[:70]}..."):
            st.markdown(f"**Summary:** {summary}")
            st.markdown(f"**Sentiment Analysis:** {sentiment}")
            if link:
                st.markdown(f"[Read full article ↗️]({link})")
    
    st.write(f"Analysis completed in {time.time()-start_time:.2f} seconds")

st.info("Note: This tool uses web scraping for some sources. Results may vary based on news availability and website structures.")
