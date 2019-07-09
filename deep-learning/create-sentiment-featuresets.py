import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer


# Lemmatizer is used to convert several words written in differents ways to the same word, e. g. ran, run -> run.
lemmatizer = WordNetLemmatizer()
hm_lines = 100000

