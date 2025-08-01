from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('model/fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    data = [news]
    vect = vectorizer.transform(data)
    prediction = model.predict(vect)
    result = prediction[0]
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)