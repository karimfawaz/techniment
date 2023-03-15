# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D
import pickle

# Load the dataset
df = pd.read_csv('train_df_2018.csv')

# Tokenize the tweets
tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
tokenized = df['tweet'].apply(
    (lambda x: tokenizer.encode(x, add_special_tokens=True)))

# Pad the sequences
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    padded, df['sentiment'], test_size=0.2, random_state=42)

# Create the model
model = Sequential()
model.add(Embedding(input_dim=tokenizer.vocab_size,
          output_dim=128, input_length=max_len))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=5, batch_size=32,
          validation_data=(X_test, y_test))

# Save the model as a pickle
pickle.dump(model, open('tweet_cnn_model.pkl', 'wb'))
