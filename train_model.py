# ✅ train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load and prepare data
df = pd.read_csv("sentiment_dataset.csv")  # Make sure this file has 'text' and 'label' columns
print("label\n", df['label'].value_counts())

texts = df['text'].astype(str).values
labels = df['label'].values

# 2. Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 3. Tokenization
max_words = 10000
max_len = 20  # ensure this is consistent in test_model.py

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# 4. Train/Test split
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42)

# 5. Build the model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=64, input_length=max_len))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# 6. Save model, tokenizer, encoder
model.save("sentiment_model.h5")
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("label_encoder.pickle", "wb") as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n✅ Model, tokenizer, and label encoder saved successfully.")
