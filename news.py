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

# Make sure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# -----------------------------
# 1) Configuration
# -----------------------------
NEWS_API_KEY = "pub_726340e45067fcad1d9a6d2fef24ba983aab3"

# Summarization model (using Pegasus as an example)
#  - You can switch to "facebook/bart-large-cnn" or "t5-small" if desired
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="google/pegasus-cnn_dailymail")

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
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
]

# -----------------------------
# 2) News Fetching
# -----------------------------

def fetch_news_newsdata(company):
    """
    Uses NewsData.io for news (title + description).
    """
    url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&q={company}&language=en&country=in&page=1"
    try:
        response = session_news.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("results", [])
        results = []
        for art in articles[:5]:
            title = art.get("title", "No Title")
            desc = art.get("description") or "No summary available"
            link = art.get("link", "")
            results.append((title, desc, link))
        return results
    except Exception:
        return []

def scrape_moneycontrol_news(company):
    """
    Scrapes headlines from MoneyControl. No summary is provided, 
    so we attempt to parse the article link for a summary if possible.
    """
    search_url = f"https://www.moneycontrol.com/news/tags/{company.replace(' ', '-').lower()}.html"
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        response = session_news.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('li', class_='clearfix')[:5]
        results = []
        for art in articles:
            h2 = art.find('h2')
            if h2 and h2.text.strip():
                title = h2.text.strip()
                link_tag = art.find('a')
                link = link_tag['href'] if link_tag else ""
                results.append((title, None, link))
        return results
    except Exception:
        return []

def scrape_bing_news(company):
    """
    Scrapes headlines from Bing News. No summary is provided,
    so we attempt to parse the article link for a summary if possible.
    """
    search_url = f"https://www.bing.com/news/search?q={company.replace(' ', '+')}"
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        response = session_news.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all("div", class_="news-card")[:5]
        results = []
        for a in articles:
            link_elem = a.find("a")
            if link_elem:
                title = link_elem.get_text(strip=True)
                link = link_elem.get("href", "")
                results.append((title, None, link))
        return results
    except Exception:
        return []

# -----------------------------
# 3) Summarize Article Body (for sources lacking a summary)
# -----------------------------
def parse_and_summarize_article(url, max_chars=2000):
    """
    Attempts to fetch the article HTML, parse text from <p> tags, 
    and generate a short summary using a summarization model.
    If blocked or no content found, returns "No summary available".
    """
    summarizer = load_summarizer()
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        resp = session_news.get(url, headers=headers, timeout=10)
        # If there's a paywall or captcha, resp may not have content
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Collect text from paragraphs
        paragraphs = soup.find_all("p")
        text_content = " ".join(p.get_text() for p in paragraphs if p.get_text())
        text_content = text_content.strip()

        # If too short or no content, fallback
        if len(text_content) < 200:
            return "No summary available"

        # Summarize only up to max_chars
        text_content = text_content[:max_chars]

        summary = summarizer(text_content, max_length=60, min_length=30, do_sample=False)
        if summary and isinstance(summary, list):
            return summary[0]["summary_text"]
        return "No summary available"
    except:
        return "No summary available"

# -----------------------------
# 4) Sentiment Analysis
# -----------------------------
def analyze_sentiment(text, method):
    """
    Uses either VADER or FinBERT to classify text.
    Returns "Positive", "Negative", or "Neutral".
    """
    try:
        if method == "VADER":
            sia = load_vader()
            compound = sia.polarity_scores(text)['compound']
            if compound >= 0.05:
                return "Positive"
            elif compound <= -0.05:
                return "Negative"
            else:
                return "Neutral"
        elif method == "FinBERT":
            finbert = load_finbert()
            result = finbert(text[:512], truncation=True)[0]
            # FinBERT typically returns 'positive', 'negative', 'neutral'
            return result['label'].capitalize()
    except:
        return "Error"
    return "Neutral"

# -----------------------------
# 5) Aggregate & Analyze News
# -----------------------------
def fetch_and_analyze_news(
    company, 
    method="VADER", 
    use_newsdata=False, 
    aggregate_sources=False
):
    """
    1. Fetch news from selected sources (NewsData.io or MoneyControl + Bing).
    2. For MoneyControl/Bing articles that have no summary, parse & summarize if possible.
    3. Analyze sentiment of headlines (or you can do headline+summary).
    4. Return overall verdict & list of (headline, summary, sentiment, link).
    """
    if aggregate_sources:
        # Combine all sources
        news_mc_bing = scrape_moneycontrol_news(company) + scrape_bing_news(company)
        news_data_io = fetch_news_newsdata(company)
        news_sources = news_data_io + news_mc_bing
    elif use_newsdata:
        # Only NewsData.io
        news_sources = fetch_news_newsdata(company)
    else:
        # Only MoneyControl + Bing
        news_sources = scrape_moneycontrol_news(company) + scrape_bing_news(company)

    if not news_sources:
        return "Neutral", []

    pos, neg, neu = 0, 0, 0
    analyzed_news = []

    for (headline, summary, link) in news_sources:
        # If summary is None, attempt to parse the link
        if not summary:
            # Summarize the actual article
            summary = parse_and_summarize_article(link)

        # Analyze sentiment (by default using HEADLINE text, 
        # but you could do headline + summary if you want more context)
        sentiment = analyze_sentiment(headline, method).capitalize()

        # Fallback if we got something weird
        if sentiment not in ["Positive", "Negative", "Neutral"]:
            sentiment = "Neutral"

        # Track counts
        if sentiment == "Positive":
            pos += 1
        elif sentiment == "Negative":
            neg += 1
        else:
            neu += 1

        analyzed_news.append((headline, summary, sentiment, link))

    total = pos + neg + neu
    if total == 0:
        return "Neutral", analyzed_news

    # Decide overall sentiment
    if (pos / total) > 0.4:
        overall = "Positive"
    elif (neg / total) > 0.4:
        overall = "Negative"
    else:
        overall = "Neutral"

    return overall, analyzed_news

# -----------------------------
# 6) Streamlit UI
# -----------------------------
st.title("📊 Stock Market News Sentiment Analysis (with Summaries)")

company = st.text_input("Enter Company Name", "HDFC Bank")

col1, col2 = st.columns(2)
with col1:
    sentiment_method = st.radio("Select Sentiment Analysis Model", ("VADER", "FinBERT"), index=0)
with col2:
    use_newsdata = st.checkbox("Use NewsData.io for news", value=False)
    aggregate_sources = st.checkbox("Aggregate Multiple Sources", value=False)

if st.button("Analyze News Sentiment"):
    with st.spinner(f"Fetching news for {company}..."):
        overall_sentiment, news_list = fetch_and_analyze_news(
            company, 
            method=sentiment_method, 
            use_newsdata=use_newsdata, 
            aggregate_sources=aggregate_sources
        )

    st.subheader(f"Overall Sentiment for {company}: **{overall_sentiment}**")

    # Build sentiment distribution
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for _, _, sentiment, _ in news_list:
        sentiment_counts[sentiment] += 1

    st.bar_chart(pd.DataFrame(sentiment_counts, index=["Sentiment"]).T)

    for headline, summary, sentiment, link in news_list:
        with st.expander(f"📰 {headline[:90]}..."):
            st.write(f"**Summary:** {summary}")
            st.write(f"**Sentiment:** {sentiment}")
            if link:
                st.markdown(f"[🔗 Read Full Article]({link})")

    st.info("If you still see 'No summary available,' it may be due to paywalls, CAPTCHAs, or missing text.")
