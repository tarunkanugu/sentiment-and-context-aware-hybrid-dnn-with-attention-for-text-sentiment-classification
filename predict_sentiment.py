import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the model
model = load_model('sentiment_model.h5')

# Function to preprocess and predict sentiment
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(padded)
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    sentiment = label_encoder.inverse_transform([predicted_label_index])[0]
    return sentiment

# Test the function
if __name__ == "__main__":
    user_input = input("Enter text to analyze sentiment: ")
    result = predict_sentiment(user_input)
    print("Predicted Sentiment:", result)
