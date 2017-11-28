# this code is authored by by Hang Jiang
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

import numpy
import pandas as pd 
from collections import Counter
import configuration as config
import model as models
import fasttext

class Preprocessing:

    def statistics(self, encoded_docs, Y, attribute):
        # find majority distribution
        result = [len(x) for x in encoded_docs]
        print("Min=%d, Mean=%d, Max=%d" % (numpy.min(result),numpy.mean(result),numpy.max(result)))
        data_max_seq_length = numpy.max(result)+1
        ct = Counter(Y)
        majorityDistribution = max([ct[k]*100/float(Y.shape[0]) for k in ct])
        print("The majority distribution is: {0} for {1}".format(majorityDistribution, attribute))
        return data_max_seq_length 

    def load_data(self, attribute):

        # read the data 
        df = pd.read_csv(config.FILENAME)
        print("The size of data is {0}".format(df.shape[0]))
        docs = df.text.astype(str).values.tolist()
        labels = df[attribute].values

        # tokenize the data 
        t = Tokenizer(num_words = config.MAX_NUM_WORDS)
        t.fit_on_texts(docs)
        encoded_docs = t.texts_to_sequences(docs)
        print("the real size of vocabulary is %d" % (len(t.word_index)+1))
        print("the truncated size of vocabulary is %d" % config.MAX_NUM_WORDS)
        self.word_index = t.word_index

        # perform Bag of Words
        data_max_seq_length = self.statistics(encoded_docs, labels, attribute) # can be used to replace MAX_SEQ_LENGTH
        self.X = sequence.pad_sequences(encoded_docs, maxlen=data_max_seq_length) # use either a fixed max-length or the real max-length from data
        self.Y = labels
        self.attribute = attribute

    def load_word_embedding(self, use_word_embedding):
        # if not use it, 
        if not use_word_embedding:
            self.embedding_matrix = []
            self.EMBEDDING_DIM = 100
            return 
        # if exist, 
        if os.path.exists("EmbeddingMatrix.pkl"):
            print "EXISTS!"
            with open("EmbeddingMatrix.pkl", 'rb') as f:
                embedding_matrix = pickle.load(f)
            self.embedding_matrix = embedding_matrix
            self.EMBEDDING_DIM = len(embedding_matrix[0])
            return 

        # read the fastText Embedding
        embeddings_index = dict()
        count_oov = 0
        f = open(config.PATH_EMBEDDING)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            EMBEDDING_DIM = len(coefs)
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))

        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((config.MAX_NUM_WORDS, EMBEDDING_DIM))
        word2idx = {k:v for k,v in self.word_index.items() if v < config.MAX_NUM_WORDS}
        for word, i in word2idx.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros
                embedding_matrix[i] = embedding_vector
            else:
                count_oov += 1
        print("Finished forming weightMatrix")
        print("%d out of %d words are NOT in the pre-trained model" % (count_oov, config.MAX_NUM_WORDS))

        # update the variables
        self.embedding_matrix = embedding_matrix
        self.EMBEDDING_DIM = EMBEDDING_DIM

        # dump the matrix 
        with open("EmbeddingMatrix.pkl", "wb") as f:
            pickle.dump(embedding_matrix, f)
        return 
    

    def load_fasttext(self, use_word_embedding):
        # if not use it, 
        if not use_word_embedding:
            self.embedding_matrix = []
            self.EMBEDDING_DIM = 100
            return 
        # if exist, 
        if os.path.exists("fastMatrix.pkl"):
            print "EXISTS!"
            with open("fastMatrix.pkl", 'rb') as f:
                embedding_matrix = pickle.load(f)
            self.embedding_matrix = embedding_matrix
            self.EMBEDDING_DIM = len(embedding_matrix[0])
            return 

        # read the fastText Embedding
        embeddings_index = dict()
        count_oov = 0
        fastModel = fasttext.load_model(config.PATH_EMBEDDING)
        EMBEDDING_DIM = len(fastModel['good'])
        print('Loaded %s word vectors.' % len(embeddings_index))

        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((config.MAX_NUM_WORDS, EMBEDDING_DIM))
        word2idx = {k:v for k,v in self.word_index.items() if v < config.MAX_NUM_WORDS}
        for word, i in word2idx.items():
            embedding_matrix[i] = fastModel[word]
            if word not in fastModel.words:
                # words not found in embedding index will be all-zeros
                count_oov += 1
        print("Finished forming weightMatrix")
        print("%d out of %d words are NOT in the pre-trained model" % (count_oov, config.MAX_NUM_WORDS))

        # update the variables
        self.embedding_matrix = embedding_matrix
        self.EMBEDDING_DIM = EMBEDDING_DIM

        # dump the matrix 
        with open("fastMatrix.pkl", "wb") as f:
            pickle.dump(embedding_matrix, f)
        return

    # generater, returns indices for train and test data  
    def cross_validation(self, X, Y, seed):
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        kfolds = kfold.split(self.X, self.Y)
        return kfolds       


def main():

    dims = ['cAGR', 'cCON', 'cEXT', 'cNEU', 'cOPN']
    choice = int(sys.argv[1])

    # preprocess set up
    preprocessObj = Preprocessing()
    paramsObj = config.Params()
    preprocessObj.load_data(dims[choice])
    preprocessObj.load_fasttext(paramsObj.use_word_embedding)

    # set seed and cross validation
    seed = 7
    numpy.random.seed(seed)
    kfolds = preprocessObj.cross_validation(preprocessObj.X, preprocessObj.Y, seed)
    cv_acc = []
    cv_records = []
    count_iter = 1

    # loop the kfolds
    for train, test in kfolds:

        # create objects for each fold of 10-fold CV
        modelObj = models.CNN()
    
        # build the model
        model = modelObj.build_simple_CNN(paramsObj=paramsObj, weight=preprocessObj.embedding_matrix)
        print(model.summary())

        # save the best model & history
        # filepath="weights.best.hdf5"
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        history = History()
        callbacks_list = [history]

        # fit the model
        model.fit(preprocessObj.X[train], preprocessObj.Y[train], 
                validation_data = (preprocessObj.X[test], preprocessObj.Y[test]),
                epochs=paramsObj.n_epoch, 
                batch_size=paramsObj.batch_size, 
                verbose=2, 
                callbacks=callbacks_list)

        # record
        ct = Counter(preprocessObj.Y)
        print("----%s: %d----" % (preprocessObj.attribute, count_iter))
        print("----highest evaluation accuracy is %f" % (100*max(history.history['val_acc'])))
        print("----dominant distribution in data is %f" % max([ct[k]*100/float(preprocessObj.Y.shape[0]) for k in ct]))
        cv_acc.append(max(history.history['val_acc']))
        cv_records.append(history.history['val_acc'])
        count_iter += 1
    
    print("The 10-fold CV score is %s" % np.nanmean(cv_acc))

if __name__ == "__main__":
    main()




































