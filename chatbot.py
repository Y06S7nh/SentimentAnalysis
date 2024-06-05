import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Ensure nltk resources are available
nltk.download('stopwords')
nltk.download('punkt')

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

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

def chatbot():
    print("Hello! I'm here to chat with you. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        sentiment = analyze_sentiment(user_input)
        print("Predicted Sentiment:", sentiment)  # Debugging statement
        if sentiment == 'positive':
            response = "I'm glad to hear that!"
        elif sentiment == 'negative':
            response = "I'm sorry to hear that."
        elif sentiment == 'neutral':
            response = "Okay. Is there anything else I can help you with?"
        print(f"Chatbot: {response}")

# Start the chatbot
chatbot()
