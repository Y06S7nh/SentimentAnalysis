from flask import Flask, request, jsonify
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os

# Ensure nltk resources are available
nltk.download('stopwords')
nltk.download('punkt')

# Print current working directory
print("Current Working Directory:", os.getcwd())

# Check if the model and vectorizer files exist
model_path = 'sentiment_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

if not os.path.isfile(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

# Load the saved model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def analyze_sentiment(user_input):
    user_input_cleaned = preprocess_text(user_input)
    user_input_tfidf = vectorizer.transform([user_input_cleaned])
    sentiment = model.predict(user_input_tfidf)[0]
    return sentiment

app = Flask(__name__)

@app.route('/')
def index():
    return 'Chatbot is running!'

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json['message']
    sentiment = analyze_sentiment(user_input)
    if sentiment == 'positive':
        response = "I'm glad to hear that!"
    elif sentiment == 'negative':
        response = "I'm sorry to hear that."
    else:
        response = "I see. How can I assist you further?"
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
