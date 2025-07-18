import random
import pickle

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

from keras.models import Sequential, load_model
from keras.layers import Input, LSTM, Dense, Activation
from keras.utils import to_categorical
from keras.optimizers import RMSprop

# Extract the data from the file
with open("modified_lines.txt", "r") as f:
    text = f.read()

# # Use the first x lines
first_x_lines = 10

# # Tokenize all words and ensure they are unique
tokenizer = RegexpTokenizer(r"[a-zA-Z']+")
tokens = tokenizer.tokenize(text.lower())

unique_tokens = np.unique(tokens)
unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}

print(len(unique_tokens))

n_words = 5
input_words = []
next_words = []

for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words])

X = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype = bool)
y = np.zeros((len(next_words), len(unique_tokens)), dtype = bool)

for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        X[i, j, unique_token_index[word]] = 1
        y[i, unique_token_index[next_words[i]]] = 1

model = Sequential()
model.add(Input(shape=(n_words, len(unique_tokens))))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation('softmax'))

model.compile(loss = "categorical_crossentropy", optimizer = RMSprop(learning_rate=0.01), metrics=['accuracy'])
model.fit(X, y, batch_size = 64, epochs = 50, shuffle = True)

model.save('message_model.keras')
model = load_model('message_model.keras')

def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    X = np.zeros((1, n_words, len(unique_tokens)))

    for i, word in enumerate(input_text.split()):
        X[0, i, unique_token_index[word]] = 1

    predictions = model.predict(X)[0]

    return np.argpartition(predictions, -n_best)[-n_best:]

possible = predict_next_word("I have decided to go", 3)

print([unique_tokens[idx] for idx in possible])