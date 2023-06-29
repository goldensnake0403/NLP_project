import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Parse JSON file into Python using json package
words = []
classes = []
documents = []
ignore_words = []
data_file =  open('intents.json').read()
intents = json.loads(data_file)

# Preprocessing data

for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        # Tokenize each word
        nltk.download('punkt')
        nltk.download('wordnet')
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        
        # add documents in the corpus
        documents.append((w, intent['tag']))
        
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
# Lemmatize, lower each word and remove duplicate
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sort classes
classes = sorted(list(set(words)))
# Output documents
print (len(documents), "documents")

print(len(classes), "classes")

print(len(words), "lemmatized words", words)

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl', 'wb')) 
        