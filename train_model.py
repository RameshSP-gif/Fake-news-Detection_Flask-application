import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# ✅ Make sure model directory exists
os.makedirs('model', exist_ok=True)

# Load dataset
df = pd.read_csv('news.csv')  # Make sure this file exists in the same folder
df = df.dropna()

X = df['text']
y = df['label']

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

with open('model/fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("✅ Model and Vectorizer saved in /model")
