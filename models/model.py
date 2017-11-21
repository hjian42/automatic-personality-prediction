import numpy as np
import glob, os, sys
import re
import pickle

import keras
from keras.layers import Dropout
from keras.datasets import imdb
from keras.models import Sequential, Model
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
from keras.layers import Merge
from keras.layers import Input

class CNN:
    
    def build_simple_CNN(self, paramsObj, weight=None):

        # Embeddings
        if weight == None or paramsObj.use_word_embedding == False:
            # NOT use word embedding
            embedding_layer = Embedding(config.MAX_NUM_WORDS, paramsObj.embeddings_dim, input_length=config.MAX_SEQ_LENGTH)
        else:
            # use word embedding
            # embedding_layer = Embedding(config.MAX_NUM_WORDS, paramsObj.embeddings_dim, input_length=config.MAX_SEQ_LENGTH)
            embedding_layer = Embedding(
                config.MAX_NUM_WORDS,
                paramsObj.embeddings_dim,
                input_length = config.MAX_SEQ_LENGTH,
                weights = [weight],
                trainable = paramsObj.train_embedding
                )

        # Convolution
        inp = Input(shape=(config.MAX_SEQ_LENGTH, paramsObj.embeddings_dim))
        conv_feature_list = []
        for filter_size, pool_size, num_filter in zip(paramsObj.filter_size, paramsObj.pool_size, paramsObj.num_filter):
            conv_layer = Conv1D(filters=num_filter, kernel_size=filter_size, strides=1, padding='same', activation='relu') (inp)
            pool_layer = MaxPooling1D(pool_size=pool_size) (conv_layer)
            flatten_layer = Flatten()(pool_layer)
            conv_feature_list.append(flatten_layer)
        out = Merge(mode='concat')(conv_feature_list)
        network = Model(input=inp, output=out)

        # Model 
        model = Sequential()
        model.add(embedding_layer)
        model.add(Dropout(paramsObj.dropout_rate))

        # convolution
        # model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
        # model.add(MaxPooling1D(pool_size=2))
        # model.add(Flatten())
        model.add(network)


        # add dense layer to complete the model
        # model.add(Dropout(paramsObj.dropout_rate))
        model.add(Dense(paramsObj.dense_layer_size, 
                    kernel_initializer='uniform', activation='softmax')) # also try 'relu'
        model.add(Dropout(paramsObj.dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model






































