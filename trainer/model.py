import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from livelossplot import PlotLossesKeras
np.random.seed(7)
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.preprocessing import sequence
# from gensim.models import Word2Vec, KeyedVectors, word2vec
import gensim
from gensim.utils import simple_preprocess
from keras.utils import to_categorical
import pickle
import h5py
from time import time
import code
import readline
import rlcompleter
from pathlib import Path
from trainer.dataset import get_mentions
from trainer.utils import interactive_console
from trainer.word2vec import get_word2vec

def train():
    dataset = get_mentions().dataset()

    word2vec = get_word2vec()



    embedding_matrix = word2vec.vectors

    top_words = embedding_matrix.shape[0]

    mxlen = 50
    nb_classes = 3

    tokenizer = Tokenizer(num_words=top_words)
    tokenizer.fit_on_texts(dataset.train_x)
    sequences_train = tokenizer.texts_to_sequences(dataset.train_x)
    sequences_test = tokenizer.texts_to_sequences(dataset.test_x)
    sequences_val = tokenizer.texts_to_sequences(dataset.val_x)

    # word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))
    # print(word_index)

    X_train = sequence.pad_sequences(sequences_train, maxlen=mxlen)
    X_test = sequence.pad_sequences(sequences_test, maxlen=mxlen)
    X_val = sequence.pad_sequences(sequences_val, maxlen=mxlen)

    Y_train = np_utils.to_categorical(dataset.train_y, nb_classes)
    Y_test = np_utils.to_categorical(dataset.test_y, nb_classes)
    Y_val = np_utils.to_categorical(dataset.val_y, nb_classes)

    # interactive_console(globals(), locals())

    batch_size = 16
    nb_epoch = 14

    print("Define embedding_layer")
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                trainable=False)

    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(128, recurrent_dropout=0.5, dropout=0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.summary()

    print("compile....")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    rnn = model.fit(X_train, Y_train, epochs=nb_epoch, batch_size=batch_size, shuffle=True, validation_data=(X_val, Y_val), verbose=1)
    score = model.evaluate(X_test, Y_test)
    print("Test Loss: %.2f%%" % (score[0] * 100))
    print("Test Accuracy: %.2f%%" % (score[1] * 100))
