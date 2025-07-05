import random
import pickle

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

text_df = pd.read_csv("fake_or_real_news.csv")
text = list(text_df.text.values)
joined_text=" ".join(text)

partial_text = joined_text[:10000]

tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_text.lower())
unique_tokens = np.unique(tokens)

print(partial_text)