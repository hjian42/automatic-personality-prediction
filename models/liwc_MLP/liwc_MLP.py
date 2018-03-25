'''
Implementation of MLP with only LIWC 2007 features on essays dataset
'''

import numpy as np
import glob, os, sys
import re
import pickle

import keras
from keras.layers import Dropout
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix

import numpy
import pandas as pd 
from collections import Counter


def cross_validation(X, Y, seed):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    kfolds = kfold.split(X, Y)
    return kfolds 

def build_MLP():
    model = Sequential()
    # model.add(Embedding(top_words, 32, input_length=max_words))
    # model.add(Flatten())
    model.add(Dense(100, activation='relu', input_shape=(80,)))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

# variables 
seed = 7
numpy.random.seed(seed)
traits = ['cAGR', 'cCON', 'cEXT', 'cOPN', 'cNEU']
num_class = 2

# initialize data
result = pd.read_csv('essays2007.csv')
X = result.loc[:,'WC':'OtherP'].values

# loop for training
for trait in traits: 
    labels = result[trait].values
    Y = keras.utils.to_categorical(labels, num_class)
    kfolds = cross_validation(X, labels, seed)
    
    cv_acc = []
    cv_records = []
    count_iter = 1
    
    for train, test in kfolds:
        history = History()
        callbacks_list = [history]
        model = build_MLP()
        model.fit(X[train], Y[train], 
                    validation_data = (X[test], Y[test]),
                    epochs=100, 
                    batch_size=64, 
                    verbose=2, 
                    callbacks=callbacks_list)
        
        # record
        ct = Counter(labels)
        print("----%s: %d----" % (trait, count_iter))
        print("----highest evaluation accuracy is %f" % (100*max(history.history['val_acc'])))
        print("----dominant distribution in data is %f" % max([ct[k]*100/float(Y.shape[0]) for k in ct]))
        cv_acc.append(max(history.history['val_acc']))
        cv_records.append(history.history['val_acc'])
        count_iter += 1
#         break
    outName = trait+".res"
    with open(outName, 'w') as out:
        out.write(trait)
        out.write("\n")
        for i, score in enumerate(cv_acc):
            out.write(str(i+1) + ': ' + str(score))
            out.write('\n')
        out.write("The Avg is %f" % np.nanmean(cv_acc))

    print("The 10-fold CV score is %s" % np.nanmean(cv_acc))










