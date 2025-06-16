import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re

MAX_SEQUENCE_LENGTH = 20

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load model and tools
model = tf.keras.models.load_model("sentiment_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Test samples
test_texts = [
    "I love this product!",
    "This is the worst experience",
    "It’s okay, nothing special.",
    "Absolutely fantastic work!",
    "Terrible, I'm never buying again",
    "Not bad, could be better"
]

preprocessed = [preprocess_text(t) for t in test_texts]
sequences = tokenizer.texts_to_sequences(preprocessed)
padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
predictions = model.predict(padded)

for text, pred in zip(test_texts, predictions):
    label = label_encoder.classes_[np.argmax(pred)]
    print(f"Text: {text} \u2794 Sentiment: {label}")
