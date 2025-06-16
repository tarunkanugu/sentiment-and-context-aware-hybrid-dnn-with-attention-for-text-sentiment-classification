from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load model
model = load_model('sentiment_model.h5')

# Preprocess and predict
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(padded)
    label_index = np.argmax(prediction, axis=1)[0]
    sentiment = label_encoder.inverse_transform([label_index])[0]
    return sentiment

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = ''
    if request.method == 'POST':
        user_input = request.form['user_input']
        sentiment = predict_sentiment(user_input)
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
