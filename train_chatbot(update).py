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
classes = sorted(list(set(classes)))
# Output documents
print (len(documents), "documents")

print(len(classes), "classes")

print(len(words), "lemmatized words", words)

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl', 'wb')) 
        
# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")
