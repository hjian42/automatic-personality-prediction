
# coding: utf-8

import pandas as pd 
import numpy as np
import glob, os, sys
import re
import pickle
import numpy
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
from collections import Counter
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History 
from sklearn.model_selection import StratifiedKFold
from keras.regularizers import l2 # L2-regularisation

def statistics(encoded_docs, Y):
    result = [len(x) for x in encoded_docs]
    print("Min=%d, Mean=%d, Max=%d" % (numpy.min(result),numpy.mean(result),numpy.max(result)))
    # global max_words
    max_w = numpy.max(result)+1
    ct = Counter(Y)
    print "The distribution of the data: ",
    print [ct[k]*100/float(Y.shape[0]) for k in ct]
    return max_w
    

def load_data(filename, personDimension):
    df = pd.read_csv(filename)
    print("The size of data is {0}".format(df.shape[0]))
    docs = df.text.astype(str).values.tolist()
    labels = df[personDimension].values
    # tokenize the data 
    t.fit_on_texts(docs)
    encoded_docs = t.texts_to_sequences(docs)
    print("the real size of vocabulary is %d" % (len(t.word_index)+1))
    print("the truncated size of vocabulary is %d" % vocab_size)
    # perform BOW and get X and Y 
    max_words = statistics(encoded_docs, labels)
    X = sequence.pad_sequences(encoded_docs, maxlen=max_words) # wrong, max_words is the max length of a sequence
    Y = labels
    return X, Y, max_words



def word_embedding(X):
    import os.path
    if os.path.exists("EmbeddingMatrix.pkl"):
        print "EXISTS!"
        with open("EmbeddingMatrix.pkl", 'rb') as f:
            embedding_matrix = pickle.load(f)
        return embedding_matrix
    embeddings_index = dict()
    f = open('../../fast-text/corpus.friends+nyt+wiki+amazon.fasttext.skip.d100.vec')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, 100))
    word2idx = {k:v for k,v in t.word_index.items() if v <= max_words}
    for word, i in word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("Finished forming weightMatrix")
    # dump the matrix 
    with open("EmbeddingMatrix.pkl", "wb") as f:
        pickle.dump(embedding_matrix, f)
    return embedding_matrix


def split_data(X, Y):
    NumOfTraining = int (0.8*len(Y))
    x_train = X[:NumOfTraining]
    x_test = X[NumOfTraining:]
    y_train = Y[:NumOfTraining]
    y_test = Y[NumOfTraining:]
    return (x_train, x_test), (y_train, y_test)

# generater, returns indices for train and test data  
def cross_validation(X, Y):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    kfolds = kfold.split(X, Y)
    return kfolds

def build_simple_CNN(dropout_rate):
    # (x_train, x_test), (y_train, y_test) = split_data(X, Y)
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_words))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(l2_lambda), activation='softmax'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, kernel_regularizer=l2(l2_lambda), activation='relu'))
    # model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # save the best model
    # filepath="weights.best.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # obtain best history
    # history = History()
    # callbacks_list = [checkpoint, history]
    # history = model.fit(X, Y, validation_split=0.2, epochs=n_epoch, batch_size=64, verbose=2, callbacks=callbacks_list)
    return model    



def build_CNN_embedding(n_epoch, dropout_rate):
    # (x_train, x_test), (y_train, y_test) = split_data(X, Y)
    model = Sequential()
    model.add(Embedding(vocab_size, 100, weights=[Matrix], input_length=max_words, trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(256, activation='softmax'))
    # model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # save the best model
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # obtain best history
    history = History()
    callbacks_list = [checkpoint, history]
    history = model.fit(X, Y, validation_split=0.2, epochs=n_epoch, batch_size=64, verbose=2, callbacks=callbacks_list)
    return history    


def build_multiple_CNN_embedding(n_epoch, dropout_rate):
    # (x_train, x_test), (y_train, y_test) = split_data(X, Y)
    model = Sequential()
    model.add(Embedding(vocab_size, 100, weights=[Matrix], input_length=max_words, trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(256, activation='softmax'))
    # model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # save the best model
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # obtain best history
    history = History()
    callbacks_list = [checkpoint, history]
    history = model.fit(X, Y, validation_split=0.2, epochs=n_epoch, batch_size=64, verbose=2, callbacks=callbacks_list)
    return history    


def plot(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.png")
    plt.clf()


# fixed variables
filename = 'data.csv'
vocab_size = 10000
n_epoch = int(sys.argv[1])
l2_lambda = 0

# cross validation
seed = 7
numpy.random.seed(seed)

# features to optimize
attributes = {'openness': 'cOPN'}
# attributes = {'neurotism': 'cNEU'}
# attributes = {'conscientiousness': 'cCON'}
# attributes = {'openness': 'cOPN', 'extraversion': 'cEXT', 'neurotism': 'cNEU', 'agreeableness': 'cAGR', 'conscientiousness': 'cCON'}
attrNames = attributes.keys()
# dropout_rates = np.arange(0.0, 1.0, 0.1)
dropout_rates = [0.5]
# l2_lambdas = [0.0001, 0.001, 0.01] # 0.01 and 0.001 do NOT give good results
attrBestScores = []

from collections import defaultdict
fiveEvals = defaultdict(list)

for attribute in attributes.values()[:]:
    t = Tokenizer(num_words = vocab_size)
    X, Y, max_words = load_data(filename, attribute)
    kfolds = cross_validation(X, Y)
    # Matrix = word_embedding(X)
    eachAttrBest = []
    for dropout_rate in dropout_rates[:]:
        cv_acc = []
        crossEvalRecord = []
        infoModel = '-'.join(['model', attribute, str(dropout_rate)])
        for train, test in kfolds:
            # create the model
            model = build_simple_CNN(dropout_rate)
            print(model.summary())
            # compile the model 
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            # save the best model
            # filepath="weights.best.hdf5"
            # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            # obtain best history
            history = History()
            callbacks_list = [history]
            # fit the model 
            history = model.fit(X[train], Y[train], validation_data=(X[test], Y[test]), epochs=n_epoch, batch_size=64, verbose=2, callbacks=callbacks_list)

            ct = Counter(Y)
            print("----%s----" % attribute)
            print("----highest evaluation accuracy is %f" % (100*max(history.history['val_acc'])))
            print("----dominant distribution in data is %f" % max([ct[k]*100/float(Y.shape[0]) for k in ct]))
            cv_acc.append(max(history.history['val_acc']))
            crossEvalRecord.append(history.history['val_acc'])
        # store crossEvalRecord into fiveEvals map
        fiveEvals[infoModel] = crossEvalRecord
        # append the cross-validation score (average)
        if np.nanmean(cv_acc):
            eachAttrBest.append(np.nanmean(cv_acc))
    if eachAttrBest:
        bestPair = (infoModel, sorted(eachAttrBest)[-1])
        attrBestScores.append(bestPair)

with open("records.txt", 'w') as out:
    for bestPair in attrBestScores:
        out.write(bestPair[0])
        print(bestPair[0])
        out.write(': ')
        out.write('%f' % bestPair[1])
        print(bestPair[1])
        out.write('\n')

with open("fiveEvals.pkl", "wb") as f:
        pickle.dump(fiveEvals, f)

# import pickle
# with open("fiveEvals.pkl", "rb") as f:
#         fiveEvals = pickle.load(f)

# print(cv_acc)
# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cv_acc)*100, numpy.std(cv_acc)*100))




# time for 10 iteration's: 10 cross*10iter*8s = 800s
# time for 10 iteration's: 10 cross*20iter*8s = 1600s

# activation function for Dense Layer: Kaixin recommends softmax instead of relu
# dense: regularizatin + dropout 
# stopwords 


