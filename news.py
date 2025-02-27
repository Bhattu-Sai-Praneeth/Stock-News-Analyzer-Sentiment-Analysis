import requests
import streamlit as st
import random
import time
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import yfinance as yf

# Download NLTK resources
nltk.download('vader_lexicon')

# List of rotating User-Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
]

# Initialize sentiment analyzers
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_finbert():
    return pipeline("text-classification", model="ProsusAI/finbert")

# Function to fetch Moneycontrol news
def scrape_moneycontrol_news(company):
    search_url = f"https://www.moneycontrol.com/news/tags/{company.replace(' ', '-').lower()}.html"
    headers = {"User-Agent": random.choice(USER_AGENTS)}

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('li', class_='clearfix')[:5]

        news_list = []
        for article in articles:
            title = article.find('h2')
            link = article.find('a', href=True)
            if title and link:
                news_list.append((title.text.strip(), link['href']))

        return news_list
    except Exception:
        return []

# Function to fetch Economic Times news
def scrape_economic_times_news(company):
    search_url = f"https://economictimes.indiatimes.com/topic/{company.replace(' ', '-')}"
    headers = {"User-Agent": random.choice(USER_AGENTS)}

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('div', class_='eachStory')[:5]

        news_list = []
        for article in articles:
            title = article.find('h3')
            link = article.find('a', href=True)
            if title and link:
                news_list.append((title.text.strip(), "https://economictimes.indiatimes.com" + link['href']))

        return news_list
    except Exception:
        return []

# Function to fetch Yahoo Finance news
def scrape_yahoo_finance_news(company):
    try:
        ticker = yf.Ticker(company)
        news = ticker.news[:5]  # Get latest 5 news articles
        return [(item['title'], item['link']) for item in news]
    except Exception:
        return []

# Function to get news from available sources
def fetch_news(company):
    news = scrape_moneycontrol_news(company)
    if not news:
        st.warning(f"Moneycontrol failed for {company}. Trying Economic Times...")
        news = scrape_economic_times_news(company)
    
    if not news:
        st.warning(f"Economic Times failed for {company}. Trying Yahoo Finance...")
        news = scrape_yahoo_finance_news(company)

    return news

# Sentiment Analysis Function
def analyze_sentiment(text, method):
    if method == "VADER":
        sia = load_vader()
        scores = sia.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    elif method == "FinBERT":
        finbert = load_finbert()
        result = finbert(text)[0]
        return result['label'].capitalize()

# Streamlit UI
st.title("News Scraper & Sentiment Analyzer")

# Sentiment analysis method selection
method = st.radio("Select Sentiment Analysis Model:", 
                 ("VADER (General Purpose)", "FinBERT (Financial Specific)"),
                 index=0)

# Extract method name from selection
method = "VADER" if "VADER" in method else "FinBERT"

# Manual company input
companies_input = st.text_area("Enter company names (comma-separated)")

# Process button
if st.button("Fetch News & Analyze"):
    companies = [c.strip() for c in companies_input.split(",") if c.strip()]
    
    if not companies:
        st.error("❌ Please enter at least one company name.")
    else:
        st.write(f"Fetching news for **{len(companies)}** companies using {method}...")

        news_data = {}
        sentiment_summary = {}

        for company in companies:
            st.write(f"Fetching news for **{company}**...")

            # Fetch news with fallback
            headlines = fetch_news(company)

            pos_count, neg_count, neu_count = 0, 0, 0
            company_news = []

            for headline, link in headlines:
                sentiment = analyze_sentiment(headline, method)
                if sentiment == "Positive":
                    pos_count += 1
                elif sentiment == "Negative":
                    neg_count += 1
                else:
                    neu_count += 1

                company_news.append([headline, sentiment, link])

            # Store news data for company
            news_data[company] = company_news

            # Compute sentiment verdict
            if pos_count > neg_count and pos_count > neu_count:
                verdict = "Positive"
            elif neg_count > pos_count and neg_count > neu_count:
                verdict = "Negative"
            else:
                verdict = "Neutral"

            sentiment_summary[company] = [pos_count, neg_count, neu_count, verdict]

            time.sleep(random.uniform(3, 6))  # Random delay between requests

        # Display News Table with Dropdowns
        if news_data:
            st.write("### News Sentiment Analysis")

            for company, articles in news_data.items():
                with st.expander(f" {company} - {len(articles)} articles"):
                    for index, (headline, sentiment, link) in enumerate(articles):
                        col1, col2, col3 = st.columns([5, 2, 2])
                        
                        with col1:
                            st.write(f"🔹 {headline}")
                        
                        with col2:
                            color = "🟩" if sentiment == "Positive" else "🟥" if sentiment == "Negative" else "⬜️"
                            st.write(f"{color} {sentiment}")
                        
                        with col3:
                            st.markdown(f'<a href="{link}" target="_blank"><button>🔗 Open Article</button></a>', unsafe_allow_html=True)

            # Sentiment Summary Table
            summary_df = pd.DataFrame.from_dict(sentiment_summary, orient='index', columns=['Positive', 'Negative', 'Neutral', 'Verdict'])
            summary_df.index.name = "Company"

            st.write("### Sentiment Summary")
            st.dataframe(summary_df, use_container_width=True)

        else:
            st.warning("⚠ No news articles found for the given companies.")
