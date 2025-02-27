import requests
import streamlit as st
import random
import time
from bs4 import BeautifulSoup
from textblob import TextBlob
import pandas as pd

# List of rotating User-Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
]

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

# Sentiment Analysis Function
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Streamlit UI
st.title("Stock News Scraper & Sentiment Analysis")

# Manual company input
companies_input = st.text_area("Enter company names (comma-separated)", "HDFCBANK, TATASTEEL, HINDALCO")

# Process button
if st.button("Fetch News & Analyze"):
    companies = [c.strip() for c in companies_input.split(",") if c.strip()]
    
    if not companies:
        st.error("❌ Please enter at least one company name.")
    else:
        st.write(f"Fetching news for **{len(companies)}** companies safely...")

        news_data = []
        sentiment_summary = {}

        for company in companies:
            st.write(f"Fetching news for **{company}**...")

            # Try Moneycontrol first
            headlines = scrape_moneycontrol_news(company)

            pos_count, neg_count, neu_count = 0, 0, 0

            for headline, link in headlines:
                sentiment = analyze_sentiment(headline)
                if sentiment == "Positive":
                    pos_count += 1
                elif sentiment == "Negative":
                    neg_count += 1
                else:
                    neu_count += 1

                news_data.append([company, headline, sentiment, link])

            # Calculate overall verdict
            if pos_count > neg_count and pos_count > neu_count:
                verdict = "Positive"
            elif neg_count > pos_count and neg_count > neu_count:
                verdict = "Negative"
            else:
                verdict = "Neutral"

            sentiment_summary[company] = [pos_count, neg_count, neu_count, verdict]

            time.sleep(random.uniform(3, 6))  # Random delay between requests

        # Display News Table
        if news_data:
            st.write("### 📰 News Sentiment Analysis")

            news_table_html = """
            <style>
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #4CAF50; color: white; }
                tr:hover { background-color: #f5f5f5; }
                a { text-decoration: none; color: blue; }
            </style>
            <table>
                <tr>
                    <th>Company</th>
                    <th>Headline</th>
                    <th>Sentiment</th>
                    <th>Link</th>
                </tr>
            """
            for row in news_data:
                news_table_html += f"""
                <tr>
                    <td>{row[0]}</td>
                    <td>{row[1]}</td>
                    <td>{row[2]}</td>
                    <td><a href="{row[3]}" target="_blank">🔗 Open Article</a></td>
                </tr>
                """
            news_table_html += "</table>"

            st.markdown(news_table_html, unsafe_allow_html=True)

            # Sentiment Summary Table
            summary_df = pd.DataFrame.from_dict(sentiment_summary, orient='index', columns=['Positive', 'Negative', 'Neutral', 'Verdict'])
            summary_df.index.name = "Company"

            st.write("### Sentiment Summary")
            st.dataframe(summary_df, use_container_width=True)  # Wider table

            # Download option
            csv_output = pd.DataFrame(news_data, columns=['Company', 'Headline', 'Sentiment', 'Link']).to_csv(index=False).encode('utf-8')
            st.download_button(label=" Download News Data", data=csv_output, file_name="news_sentiment_analysis.csv", mime="text/csv")

            summary_csv = summary_df.to_csv(index=True).encode('utf-8')
            st.download_button(label=" Download Sentiment Summary", data=summary_csv, file_name="sentiment_summary.csv", mime="text/csv")

        else:
            st.warning("⚠ No news articles found for the given companies.")
