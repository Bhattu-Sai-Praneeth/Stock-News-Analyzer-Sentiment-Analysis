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

# Updated scraping functions
def scrape_moneycontrol_news(company):
    search_url = f"https://www.moneycontrol.com/news/tags/{company.replace(' ', '-').lower()}.html"
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.moneycontrol.com"
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('li', class_='clearfix')[:5]

        return [(article.find('h2').text.strip(), article.find('a')['href'])
                for article in articles if article.find('h2') and article.find('a')]
    except Exception as e:
        st.error(f"Moneycontrol Error: {str(e)}")
        return []

def scrape_economic_times_news(company):
    search_url = f"https://economictimes.indiatimes.com/topic/{company.replace(' ', '%20')}"
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('div', {'class': 'story_list'})[:5]

        return [(article.find('h3').text.strip(), 
                f"https://economictimes.indiatimes.com{article.find('a')['href']}")
                for article in articles if article.find('h3') and article.find('a')]
    except Exception as e:
        st.error(f"Economic Times Error: {str(e)}")
        return []

def scrape_yahoo_finance_news(company):
    try:
        ticker = yf.Ticker(company)
        news = ticker.news[:5]
        return [(item['title'], item['link']) for item in news if 'title' in item and 'link' in item]
    except Exception as e:
        st.error(f"Yahoo Finance Error: {str(e)}")
        return []

def fetch_news(company):
    sources = [
        ("Moneycontrol", scrape_moneycontrol_news),
        ("Economic Times", scrape_economic_times_news),
        ("Yahoo Finance", scrape_yahoo_finance_news)
    ]
    
    for source_name, scraper in sources:
        news = scraper(company)
        if news:
            return news
        st.warning(f"{source_name} failed for {company}. Trying next source...")
        time.sleep(2)
    
    return []

# Sentiment Analysis Function
def analyze_sentiment(text, method):
    try:
        if method == "VADER":
            sia = load_vader()
            scores = sia.polarity_scores(text)
            compound = scores['compound']
            if compound >= 0.05:
                return "Positive"
            elif compound <= -0.05:
                return "Negative"
            return "Neutral"
        elif method == "FinBERT":
            finbert = load_finbert()
            result = finbert(text[:512], truncation=True)[0]  # Truncate to 512 tokens
            return result['label'].capitalize()
    except Exception as e:
        st.error(f"Sentiment Analysis Error: {str(e)}")
        return "Error"

# Streamlit UI
st.title("News Scraper & Sentiment Analyzer")

# Input improvements
st.markdown("**Enter company names or stock tickers (e.g., 'Apple' or 'AAPL'):**")
companies_input = st.text_area("Separate multiple entries with commas", 
                             placeholder="Apple, AAPL, Tata Motors")

method = st.radio("Sentiment Analysis Model:", 
                 ("VADER (General Purpose)", "FinBERT (Financial Specific)"),
                 index=1).split()[0]

if st.button("Fetch News & Analyze"):
    companies = [c.strip() for c in companies_input.split(",") if c.strip()]
    
    if not companies:
        st.error("Please enter at least one company name/ticker")
    else:
        progress_bar = st.progress(0)
        news_data = {}
        sentiment_summary = {}

        for idx, company in enumerate(companies):
            progress = (idx + 1) / len(companies)
            progress_bar.progress(progress)
            
            st.subheader(f"Analyzing: {company}")
            with st.spinner(f"Fetching news for {company}..."):
                news = fetch_news(company)
            
            if not news:
                st.warning(f"No news found for {company}")
                continue

            # Process news
            company_news = []
            pos = neg = neu = 0
            
            for headline, link in news:
                sentiment = analyze_sentiment(headline, method)
                company_news.append((headline, sentiment, link))
                
                if sentiment == "Positive": pos += 1
                elif sentiment == "Negative": neg += 1
                else: neu += 1

            # Determine overall sentiment
            total = pos + neg + neu
            verdict = "Neutral"
            if total > 0:
                if pos/total > 0.4: verdict = "Positive"
                elif neg/total > 0.4: verdict = "Negative"

            # Store results
            news_data[company] = company_news
            sentiment_summary[company] = {
                "Positive": pos,
                "Negative": neg,
                "Neutral": neu,
                "Verdict": verdict
            }

            # Display results
            st.success(f"Analyzed {len(news)} articles - Overall Sentiment: {verdict}")
            time.sleep(1)  # Visual progress spacing

        # Display final results
        st.subheader("Final Analysis Report")
        
        if news_data:
            # Sentiment Summary
            st.write("### Sentiment Summary")
            summary_df = pd.DataFrame.from_dict(sentiment_summary, orient='index')
            st.dataframe(summary_df.style.highlight_max(axis=0, color='#90EE90')
            
            # Detailed News View
            st.write("### Detailed News Analysis")
            for company, articles in news_data.items():
                with st.expander(f"{company} ({len(articles)} articles)"):
                    for headline, sentiment, link in articles:
                        st.markdown(f"""
                        **{headline}**  
                        Sentiment: `{sentiment}`  
                        [Read Article]({link})  
                        """)
        else:
            st.error("No news found across all sources for the given companies")

        progress_bar.empty()
