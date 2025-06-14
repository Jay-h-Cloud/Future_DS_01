# --- Import Libraries ---
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import tweepy
from bs4 import BeautifulSoup
import requests
import time

# Twitter API credentials - Replace with your own
# consumer_key = "YOUR_CONSUMER_KEY"
# consumer_secret = "YOUR_CONSUMER_SECRET"
# access_token = "YOUR_ACCESS_TOKEN"
# access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

# # Initialize Twitter API
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)
# api = tweepy.API(auth, wait_on_rate_limit=True)

# Function to get tweets
# def get_tweets(query, count=100):
#     tweets = []
#     try:
#         for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="en").items(count):
#             tweets.append(tweet.text)
#     except Exception as e:
#         print(f"Error fetching tweets: {e}")
#     return tweets

# Function to scrape web content
def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Example: getting all paragraph texts
        texts = [p.text for p in soup.find_all('p')]
        return ' '.join(texts)
    except Exception as e:
        print(f"Error scraping website: {e}")
        return ""

# Load existing dataset
file_path = "C:/Users/jayde/Downloads/sentimentdataset.csv"
df = pd.read_csv(file_path)

# Add scraped data (example)
# Get tweets related to common sentiment-related topics
# search_terms = ["customer service", "product review", "brand experience", "shopping experience"]
# tweets = []
# for term in search_terms:
#     tweets.extend(get_tweets(term, count=25))  # 25 tweets per topic = 100 total

# Scrape reviews/feedback from major review websites
websites = [
    "https://www.trustpilot.com/categories/shopping_retail",
    "https://www.consumeraffairs.com/online/",
    "https://www.sitejabber.com/categories/online-shopping"
]
web_text = ""
for url in websites:
    web_text += scrape_website(url) + " "

# Create DataFrame from scraped data
# scraped_df = pd.DataFrame({
#     'Text': tweets + [web_text]
# })

# Combine existing and scraped data
# df = pd.concat([df, scraped_df], ignore_index=True)

# --- Clean Text ---
def clean_text(text):
    text = re.sub(r"http\S+", "", text)              # Remove URLs
    text = re.sub(r"#\S+", "", text)                 # Remove hashtags
    text = re.sub(r"@\S+", "", text)                 # Remove mentions
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)      # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()         # Remove extra spaces
    return text.lower()                                # Convert to lowercase

df['Cleaned_Text'] = df['Text'].apply(clean_text)

# --- Remove Stopwords ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df['Cleaned_Text'] = df['Cleaned_Text'].apply(
    lambda x: " ".join([word for word in x.split() if word not in stop_words])
)

# --- Lemmatization ---
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
df['Cleaned_Text'] = df['Cleaned_Text'].apply(
    lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()])
)

# --- Sentiment Analysis ---
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

df['Predicted_Sentiment'] = df['Cleaned_Text'].apply(get_sentiment)

# --- TF-IDF Keyword Extraction ---
tfidf = TfidfVectorizer(max_features=5, stop_words='english')
X = tfidf.fit_transform(df['Cleaned_Text'].astype(str))
keywords = tfidf.get_feature_names_out()

def get_top_keywords(row):
    vector = tfidf.transform([row])
    indices = vector.toarray().argsort()[0][-5:][::-1]
    return ', '.join([keywords[i] for i in indices if i < len(keywords)])

df['Keywords/Topics'] = df['Cleaned_Text'].astype(str).apply(get_top_keywords)

# --- Optional: Preview the Processed Data ---
print("DataFrame head after processing:")
print(df.head())
print("\nAvailable Columns:")
print(df.columns)

# --- Export the Cleaned Data to CSV (For Power BI or further analysis) ---
export_path = "C:/Users/jayde/Downloads/powerbi.csv"
df.to_csv(export_path, index=False)
print(f"Cleaned data exported to: {export_path}")

