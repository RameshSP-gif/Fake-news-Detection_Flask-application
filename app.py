from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer
with open('model/fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    data = [news]
    vect = vectorizer.transform(data)
    prediction = model.predict(vect)
    result = prediction[0]
    return render_template('index.html', prediction=result)

# Run the app with dynamic port and host
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # default to 5000 for local testing
    app.run(host='0.0.0.0', port=port)
