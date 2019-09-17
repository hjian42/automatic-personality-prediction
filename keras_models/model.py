import keras
import glob, os, sys, re, pickle
import pandas as pd 
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, TimeDistributed
from keras.layers import Dropout, Activation , LSTM, Bidirectional, GRU
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from collections import Counter
import configuration as config
from keras.layers import Input, BatchNormalization
from keras.layers.merge import concatenate

from attention import AttLayer

class DeepModel:

    def chooseModel(self, modelName, paramsObj, weight=[]):
        models = {  
                    'MLP': self.build_MLP,
                    'textCNN': self.build_textCNN,
                    'BLSTM': self.build_BLSTM,
                    'CNN': self.build_LeNet5,
                    'C-LSTM': self.build_CLSTM,
                    'BiGRU': self.build_BiGRU,
                    'Attention': self.build_ABLSTM, # sentence-level ABLSTM
                    'HAN': self.build_HAN1, 
                    'ABCNN': self.build_hang
                  }

        return models[modelName](paramsObj, weight)

    # two implementatins of HAN because 1. synthesio's implementation 2. richard's implementation
    # richard's implementation: let it run, check acc
    def build_HAN1(self, paramsObj, weight=[]):

        # Embeddings
        if len(weight) == 0 or paramsObj.use_word_embedding == False:
            # NOT use word embedding
            embedding_layer = Embedding(config.MAX_NUM_WORDS, config.EMBEDDING_DIM, input_length=config.MAX_SEQ_LENGTH)
        else:
            # use word embedding
            embedding_layer = Embedding(
                config.MAX_NUM_WORDS,
                config.EMBEDDING_DIM,
                input_length = config.MAX_SEQ_LENGTH,
                weights = [weight],
                trainable = paramsObj.train_embedding
                )

        # Create the sentModel 
        sentence_input = Input(shape=(config.MAX_SEQ_LENGTH, ), # no need to specify the last dimension, why
                       dtype='int32',
                       name='sentence_input')
        embedding_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedding_sequences)
        l_dense = TimeDistributed(Dense(200))(l_lstm)
        l_att = AttLayer()(l_dense)
        sentEncoder = Model(sentence_input, l_att)

        # dialogModel
        dialog_input = Input(shape=(config.MAX_SENTS, config.MAX_SEQ_LENGTH), dtype='int32')
        dialog_encoder = TimeDistributed(sentEncoder)(dialog_input)
        l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(dialog_encoder)
        l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
        l_att_sent = AttLayer()(l_dense_sent)

        # output layer
        preds = Dense(2, activation='softmax')(l_att_sent)
        model = Model(dialog_input, preds)

        model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

        return model

    def build_hang(self, paramsObj, weight=[]):

        # Embeddings
        if len(weight) == 0 or paramsObj.use_word_embedding == False:
            # NOT use word embedding
            embedding_layer = Embedding(config.MAX_NUM_WORDS, config.EMBEDDING_DIM, input_length=config.MAX_SEQ_LENGTH)
        else:
            # use word embedding
            embedding_layer = Embedding(
                config.MAX_NUM_WORDS,
                config.EMBEDDING_DIM,
                input_length = config.MAX_SEQ_LENGTH,
                weights = [weight],
                trainable = paramsObj.train_embedding
                )

        # Create Model 
        main_input = Input(shape=(config.MAX_SEQ_LENGTH, ), # no need to specify the last dimension, why
                       dtype='int32',
                       name='main_input')
        embedding_sequences = embedding_layer(main_input)
        embedding_sequences = Dropout(paramsObj.dropout_rate)(embedding_sequences)

        conv_feature_list = []
        for filter_size, pool_size, num_filter in zip(paramsObj.filter_size, paramsObj.pool_size, paramsObj.num_filter):
            conv_layer = Conv1D(filters=num_filter, kernel_size=filter_size, strides=1, padding='same', activation='relu') (embedding_sequences)
            pool_layer = MaxPooling1D(pool_size=pool_size) (conv_layer)
            conv_feature_list.append(pool_layer)
        if (len(conv_feature_list) == 1):
            out =  conv_feature_list[0]
        else:
            out = concatenate(conv_feature_list, axis=1)
        # network = Model(inputs=cnn_inp, outputs=out)
        
        X = TimeDistributed(Dense(len(paramsObj.filter_size)*paramsObj.pool_size[0]), name='DenseTimeDistributed')(out)
        X = AttLayer(name='AttLayer')(X)

        # add dense layer to complete the model
        X = Dropout(paramsObj.dropout_rate)(X)
        X = Dense(paramsObj.dense_layer_size, 
                  kernel_initializer='uniform', 
                  activation='relu')(X)

        # output layer
        predictions = Dense(config.ClassNum, activation='softmax')(X)
        model = Model(main_input, predictions)

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        return model

    def build_ABCNN(self, paramsObj, weight=[]):

        # Embeddings
        if len(weight) == 0 or paramsObj.use_word_embedding == False:
            # NOT use word embedding
            embedding_layer = Embedding(config.MAX_NUM_WORDS, config.EMBEDDING_DIM, input_length=config.MAX_SEQ_LENGTH)
        else:
            # use word embedding
            embedding_layer = Embedding(
                config.MAX_NUM_WORDS,
                config.EMBEDDING_DIM,
                input_length = config.MAX_SEQ_LENGTH,
                weights = [weight],
                trainable = paramsObj.train_embedding
                )

        # Create Model 
        main_input = Input(shape=(config.MAX_SEQ_LENGTH,),
                       dtype='int32',
                       name='main_input')
        embedding_sequences = embedding_layer(main_input)

        # params 
        inner = 'outer'
        type = 'atten'


        if(inner=='inner'):
            padding='valid'
        else:
            padding='same'

        conv_att_features = []
        # i did not use the pool_size and num_filter here
        nb_filter = 10
        for filter_length, pool_size, num_filter in zip(paramsObj.filter_size, paramsObj.pool_size, paramsObj.num_filter):
            convolution_layer = Conv1D(
                filters=nb_filter,
                kernel_size=filter_length,
                padding =padding,
                activation='relu',
                name='convLayer'+str(filter_length)
            )
            conv_out = convolution_layer(embedding_sequences)

            ###attenton#########
            if(type=='atten' and inner=='inner'):
                att_inpt = TimeDistributed(Dense(nb_filter))(conv_out)
                att_out = AttLayer(name='AttLayer'+str(filter_length))(att_inpt)
                conv_att_features.append(att_out)
            elif(type=='max'):
                out = MaxPooling1D(
                    name='maxPooling'+ str(filter_length),
                    pool_size=(config.MAX_SEQ_LENGTH - filter_length + 1)
                )(conv_out)
                conv_att_features.append(out)
            else:
                conv_att_features.append(conv_out)


        if(len(paramsObj.filter_size)>1):
            X = concatenate(conv_att_features, axis=1)
        else:
            X = conv_att_features[0]

        if(type=='max'):
            X = Flatten()(X)
        if(inner=='outer'):
            X = TimeDistributed(Dense(len(paramsObj.filter_size)*nb_filter),name='DenseTimeDistributed')(X)
            X = AttLayer(name='AttLayer')(X)

        X = Dropout(0.9)(X)

        # x = Dense(output_dim=hidden_dims, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01))(attention_features)
        hidden_dims = 100
        x = Dense(units=hidden_dims,activation='relu')(X)

        # dense hidden layer
        predictions = Dense(config.ClassNum, activation='softmax')(x)

        # build the model
        model = Model(main_input, predictions)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        return model

    def build_ABLSTM(self, paramsObj, weight=[]):

        model = Sequential()

        # Embeddings
        if len(weight) == 0 or paramsObj.use_word_embedding == False:
            model.add(Embedding(config.MAX_NUM_WORDS,config.EMBEDDING_DIM,input_length=config.MAX_SEQ_LENGTH))
        else:
            model.add(Embedding(
                config.MAX_NUM_WORDS,
                config.EMBEDDING_DIM,
                weights=[weight],
                input_length=config.MAX_SEQ_LENGTH,
                trainable=paramsObj.train_embedding))

        model.add(Bidirectional(GRU(128, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
        # TODO: add time steps again

        model.add(AttLayer())

        model.add(Dense(config.ClassNum, activation='softmax'))
        model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

        return model


    def build_CLSTM(self, paramsObj, weight=[]):

        # Embeddings
        if len(weight) == 0 or paramsObj.use_word_embedding == False:
            # NOT use word embedding
            embedding_layer = Embedding(config.MAX_NUM_WORDS, config.EMBEDDING_DIM, input_length=config.MAX_SEQ_LENGTH)
        else:
            # use word embedding
            embedding_layer = Embedding(
                config.MAX_NUM_WORDS,
                config.EMBEDDING_DIM,
                input_length = config.MAX_SEQ_LENGTH,
                weights = [weight],
                trainable = paramsObj.train_embedding
                )

         # Model 
        model = Sequential()
        model.add(embedding_layer)

        # Convolution
        model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2, padding='same'))

        # LSTM
        model.add(Bidirectional(GRU(256, dropout=paramsObj.dropout_rate, recurrent_dropout=0.1)))
        model.add(Dropout(paramsObj.dropout_rate))

        # classifier layer
        # model.add(Dense(config.ClassNum, activation='softmax'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

         # classifier layer 
        if not config.multilabel:
            model.add(Dense(config.ClassNum, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            model.add(Dense(config.ClassNum, activation='sigmoid'))
            # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model



    def build_BLSTM(self, paramsObj, weight=[]):

        model = Sequential()

        # Embeddings
        if len(weight) == 0 or paramsObj.use_word_embedding == False:
            # which vocab size to use word_index or max_vocab+1 ???
            model.add(Embedding(config.MAX_NUM_WORDS,config.EMBEDDING_DIM,input_length=config.MAX_SEQ_LENGTH))
        else:
            model.add(Embedding(
                config.MAX_NUM_WORDS,
                config.EMBEDDING_DIM,
                weights=[weight],
                input_length=config.MAX_SEQ_LENGTH,
                trainable=paramsObj.train_embedding))

        # model.add(Bidirectional(LSTM(64,return_sequences=True),merge_mode=paramsObj.merge_mode))
        # model.add(Dropout(paramsObj.dropout_rate))
        # model.add(Bidirectional(LSTM(128,return_sequences=True),merge_mode=paramsObj.merge_mode))
        # model.add(Dropout(paramsObj.dropout_rate))
        model.add(Bidirectional(LSTM(128)))
        # model.add(Dense(paramsObj.dense_layer_size, 
        #             kernel_initializer='uniform', activation='softmax'))
        model.add(Dropout(paramsObj.dropout_rate))
        model.add(Dense(config.ClassNum, activation='sigmoid'))
        adam = Adam(lr=0.0001, decay=1e-5)
        model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

        return model

    def build_BiGRU(self, paramsObj, weight=[]):

        model = Sequential()

        # Embeddings
        if len(weight) == 0 or paramsObj.use_word_embedding == False:
            # which vocab size to use word_index or max_vocab+1 ???
            model.add(Embedding(config.MAX_NUM_WORDS,config.EMBEDDING_DIM,input_length=config.MAX_SEQ_LENGTH))
        else:
            model.add(Embedding(
                config.MAX_NUM_WORDS,
                config.EMBEDDING_DIM,
                weights=[weight],
                input_length=config.MAX_SEQ_LENGTH,
                trainable=paramsObj.train_embedding))

        # model.add(Bidirectional(LSTM(64,return_sequences=True),merge_mode=paramsObj.merge_mode))
        # model.add(Dropout(paramsObj.dropout_rate))
        # model.add(Bidirectional(GRU(128,return_sequences=True),merge_mode=paramsObj.merge_mode))
        # model.add(Dropout(paramsObj.dropout_rate))
        model.add(Bidirectional(GRU(128)))
        # model.add(Dense(paramsObj.dense_layer_size, 
        #             kernel_initializer='uniform', activation='softmax'))
        model.add(Dropout(paramsObj.dropout_rate))
        model.add(Dense(config.ClassNum, activation='softmax'))
        adam = Adam(lr=0.0001, decay=1e-5)
        model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

        return model


    def build_LeNet5(self, paramsObj, weight=[]):

        # Embeddings
        if len(weight) == 0 or paramsObj.use_word_embedding == False:
            # NOT use word embedding
            embedding_layer = Embedding(config.MAX_NUM_WORDS, config.EMBEDDING_DIM, input_length=config.MAX_SEQ_LENGTH)
        else:
            # use word embedding
            embedding_layer = Embedding(
                config.MAX_NUM_WORDS,
                config.EMBEDDING_DIM,
                input_length = config.MAX_SEQ_LENGTH,
                weights = [weight],
                trainable = paramsObj.train_embedding
                )

        # Model 
        model = Sequential()
        model.add(embedding_layer)

        # Convolution
        model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=3, padding='same'))
        model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=3, padding='same'))
        model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
        model.add(Flatten())

        # add dense layer to complete the model
        model.add(Dropout(paramsObj.dropout_rate))
        model.add(BatchNormalization())
        model.add(Dense(paramsObj.dense_layer_size, 
                    kernel_initializer='uniform', activation='relu')) # or softmax
        model.add(Dropout(paramsObj.dropout_rate))

        # # classifier layer
        # model.add(Dense(config.ClassNum, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # classifier layer 
        if not config.multilabel:
            model.add(Dense(config.ClassNum, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            model.add(Dense(config.ClassNum, activation='sigmoid'))
            # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model



    def build_textCNN(self, paramsObj, weight=[]):

        # Embeddings
        if len(weight) == 0 or paramsObj.use_word_embedding == False:
            # NOT use word embedding
            embedding_layer = Embedding(config.MAX_NUM_WORDS, config.EMBEDDING_DIM, input_length=config.MAX_SEQ_LENGTH)
        else:
            # use word embedding
            embedding_layer = Embedding(
                config.MAX_NUM_WORDS,
                config.EMBEDDING_DIM,
                input_length = config.MAX_SEQ_LENGTH,
                weights = [weight],
                trainable = paramsObj.train_embedding
                )

        # Create Model 
        model = Sequential()
        model.add(embedding_layer)

        # Convolution
        inp = Input(shape=(config.MAX_SEQ_LENGTH, config.EMBEDDING_DIM))
        conv_feature_list = []
        for filter_size, pool_size, num_filter in zip(paramsObj.filter_size, paramsObj.pool_size, paramsObj.num_filter):
            conv_layer = Conv1D(filters=num_filter, kernel_size=filter_size, strides=1, padding='same', activation='relu') (inp)
            pool_layer = MaxPooling1D(pool_size=pool_size, strides=pool_size, padding='same') (conv_layer)
            flatten_layer = Flatten()(pool_layer)
            conv_feature_list.append(flatten_layer)
        if (len(conv_feature_list) == 1):
            out =  conv_feature_list[0] 
        else:
            out = concatenate(conv_feature_list, axis=-1)
        network = Model(inputs=inp, outputs=out)
        model.add(network)
       
        # add dense layer to complete the model
        model.add(Dropout(paramsObj.dropout_rate))
        model.add(BatchNormalization()) # ???
        model.add(Dense(paramsObj.dense_layer_size, 
                    kernel_initializer='uniform', activation='softmax')) # also try 'relu'
        model.add(Dropout(paramsObj.dropout_rate))
        
        # classifier layer 
        if not config.multilabel:
            model.add(Dense(config.ClassNum, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            model.add(Dense(config.ClassNum, activation='sigmoid'))
            # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model


    def build_MLP(self, paramsObj, weight=[]):

        model = Sequential()

        # Embeddings
        if len(weight) == 0 or paramsObj.use_word_embedding == False:
            embedding_layer = Embedding(config.MAX_NUM_WORDS, config.EMBEDDING_DIM, input_length=config.MAX_SEQ_LENGTH)
        else:
            embedding_layer = Embedding(
                config.MAX_NUM_WORDS,
                config.EMBEDDING_DIM,
                input_length = config.MAX_SEQ_LENGTH,
                weights = [weight],
                trainable = paramsObj.train_embedding
                )
        model.add(embedding_layer)

        # Dense Layer and dropout
        model.add(Dense(512, input_shape=(config.MAX_SEQ_LENGTH,config.EMBEDDING_DIM), activation='relu'))
        model.add(Dropout(paramsObj.dropout_rate))
        model.add(Flatten())

        # classifier layer
        # model.add(Dense(config.ClassNum, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', 
        #               optimizer='adam', 
        #               metrics=['accuracy'])

         # classifier layer 
        if not config.multilabel:
            model.add(Dense(config.ClassNum, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            model.add(Dense(config.ClassNum, activation='sigmoid'))
            # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        

        return model

    

  
