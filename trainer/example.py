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
from gensim.models import Word2Vec, KeyedVectors, word2vec
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

def copen(_globals, _locals):
    """
    Opens interactive console with current execution state.
    Call it with: `console.open(globals(), locals())`
    """
    context = _globals.copy()
    context.update(_locals)
    readline.set_completer(rlcompleter.Completer(context).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(context)
    shell.interact()


DATA_FILE = ''
WORD_EMBEDDINGS_FILE = ''
OUTPUT_DIR = ''


def train_and_evaluate():

    dataset = pd.read_csv(DATA_FILE, delimiter = ",")

    # Delete unused column
    del dataset['length']

    # Delete All NaN values from columns=['description','rate']
    dataset = dataset[dataset['description'].notnull() & dataset['rate'].notnull()]

    # We set all strings as lower case letters
    dataset['description'] = dataset['description'].str.lower()

    X = dataset['description']
    Y = dataset['rate']
    X = X[0:1000]
    Y = Y[0:1000]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    print("X_train shape: " + str(X_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("X_val shape: " + str(X_val.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("Y_test shape: " + str(Y_test.shape))
    print("Y_val shape: " + str(Y_val.shape))


    s = time()
    print("Loading words embedding...")
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(WORD_EMBEDDINGS_FILE, binary=False)
    print("Done loading in {}".format(time() - s))
    embedding_matrix = word2vec_model.vectors
    print('Shape of embedding matrix: ', embedding_matrix.shape)

    top_words = embedding_matrix.shape[0]

    mxlen = 50
    nb_classes = 3

    tokenizer = Tokenizer(num_words=top_words)
    tokenizer.fit_on_texts(X_train)
    sequences_train = tokenizer.texts_to_sequences(X_train)
    sequences_test = tokenizer.texts_to_sequences(X_test)
    sequences_val = tokenizer.texts_to_sequences(X_val)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    print(word_index)

    X_train = sequence.pad_sequences(sequences_train, maxlen=mxlen)
    X_test = sequence.pad_sequences(sequences_test, maxlen=mxlen)
    X_val = sequence.pad_sequences(sequences_val, maxlen=mxlen)

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_val = np_utils.to_categorical(Y_val, nb_classes)

    batch_size = 32
    nb_epoch = 12

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
    rnn = model.fit(X_train, Y_train, epochs= 1, batch_size=batch_size, shuffle=True, validation_data=(X_val, Y_val), verbose=2)
    score = model.evaluate(X_val, Y_val)
    print("Test Loss: %.2f%%" % (score[0]*100))
    print("Test Accuracy: %.2f%%" % (score[1]*100))

    path = Path(OUTPUT_DIR)
    path.mkdir(parents=True, exist_ok=True)

    print('Save model...')
    model.save(OUTPUT_DIR+'/finalsentimentmodel.h5')
    print('Saved model to disk...')

    print('Save Word index...')

    output = open(OUTPUT_DIR + "/model.h5", 'wb')
    pickle.dump(word_index, output)
    print('Saved word index to disk...')
