# --- Import Libraries ---
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Load Dataset ---
file_path = "C:/Users/jayde/Downloads/sentimentdataset.csv"
df = pd.read_csv(file_path)

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
export_path = "C:/Users/jayde/Downloads/cleaned_sentiment_data_step2.csv"
df.to_csv(export_path, index=False)
print(f"Cleaned data exported to: {export_path}")

