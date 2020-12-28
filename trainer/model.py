import json
import os
import typing

import numpy as np

np.random.seed(7)
from keras.preprocessing.text import Tokenizer
from keras.models import load_model, Sequential
from keras.layers import Dense, Activation, LSTM, Masking, Input
from keras.utils import np_utils
from tabulate import tabulate
from pathlib import Path

from trainer.api import MentionExpanded
import trainer.word2vector


class Model(object):
    __sentiment2int = {
        'POSITIVE': 2,
        'NEUTRAL': 1,
        'NEGATIVE': 0
    }
    __int2sentiment = {v: k for k, v in __sentiment2int.items()}

    def __init__(self, word2vector: trainer.word2vector.Word2Vector):
        self.word2vector = word2vector
        self.model = None
        self.parameters = {
            'maxlen': 32,
            'epochs': 44,
            'batch_size': 32
        }

    def train(self,
              train_mentions,
              val_mentions=None,
              epochs=None,
              batch_size=None,
              maxlen=None,
              verbose=1
              ):
        self.parameters['maxlen'] = maxlen or self.parameters['maxlen']
        self.parameters['epochs'] = epochs or self.parameters['epochs']
        self.parameters['batch_size'] = batch_size or self.parameters['batch_size']

        # 1. Define model
        self.model = Sequential()
        self.model.add(Input(shape=(None, 300)))
        self.model.add(Masking(mask_value=self.word2vector.default_vector))
        self.model.add(LSTM(128, recurrent_dropout=0.5, dropout=0.5, input_shape=(None, 300)))
        self.model.add(Dense(3))
        self.model.add(Activation('softmax'))

        self.model.summary()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # 2. train model
        positive_train_mentions = list(filter(lambda x: x.sentiment == 'POSITIVE', train_mentions))
        train_mentions = positive_train_mentions + train_mentions
        train_x, train_y = self._mentions_to_xs(train_mentions), self._mentions_to_ys(train_mentions)

        val_xy = (self._mentions_to_xs(val_mentions), self._mentions_to_ys(val_mentions)) if val_mentions else ()

        self.model.fit(train_x, train_y,
                       epochs=self.parameters['epochs'],
                       batch_size=self.parameters['batch_size'],
                       shuffle=True,
                       validation_data=val_xy,
                       verbose=verbose)

    def evaluate(self, mentions):
        if self.model is None:
            raise Exception("You have to train model first.")

        test_x, test_y = self._mentions_to_xs(mentions), self._mentions_to_ys(mentions)

        score = self.model.evaluate(test_x, test_y)
        print("Test Loss: %.2f%%" % (score[0] * 100))
        print("Test Accuracy: %.2f%%" % (score[1] * 100))

        test_y_true = [np.argmax(i) for i in test_y]
        test_y_predicted = [np.argmax(i) for i in self.model.predict(test_x)]
        summary = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i, y_true in enumerate(test_y_true):
            summary[int(y_true)][int(test_y_predicted[i])] += 1

        def table_cell(i, j):
            return "%.2f%%" % (100 * summary[i][j] / sum(summary[i]))

        print(tabulate([
            ['Positive', table_cell(2, 2), table_cell(2, 1), table_cell(2, 0)],
            ['Neutral', table_cell(1, 2), table_cell(1, 1), table_cell(1, 0)],
            ['Negative', table_cell(0, 2), table_cell(0, 1), table_cell(0, 0)]
        ], headers=['', 'Positive', 'Neutral', 'Negative']))

    def predict(self, mentions) -> typing.List[str]:
        if self.model is None:
            raise Exception("You have to train model first.")
        predict_x = self._mentions_to_xs(mentions)
        predicted = [np.argmax(i) for i in self.model.predict(predict_x)]

        positive_count = len(list(filter(lambda x: x == 2, predicted)))
        neutral_count = len(list(filter(lambda x: x == 1, predicted)))
        negative_count = len(list(filter(lambda x: x == 0, predicted)))
        total = positive_count + neutral_count + negative_count

        print(tabulate([
            ['Positive', positive_count, "%.2f%%" % (100 * positive_count / total)],
            ['Neutral', neutral_count, "%.2f%%" % (100 * neutral_count / total)],
            ['Negative', negative_count, "%.2f%%" % (100 * negative_count / total)]
        ], headers=['Sentiment', 'Count', '%']))
        return [self.__int2sentiment[i] for i in predicted]

    def save(self, directory: str):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.save(directory + "/model")
        with open(directory + "/parameters.json", 'w') as handle:
            json.dump(self.parameters, handle)

    def load(self, directory: str):
        if self.model is not None:
            return
        if not os.path.exists(directory + "/model"):
            raise Exception("Train and save model first.")
        self.model = load_model(directory + "/model")
        with open(directory + "/parameters.json", 'r') as handle:
            self.parameters = json.load(handle)

    def _mentions_to_xs(self, mentions: typing.List[MentionExpanded]):
        maxlen = self.parameters['maxlen']
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([m.anonymous_comment_content() for m in mentions])

        def index2vector(index):
            if index == 0:
                return self.word2vector.default_vector
            return self.word2vector.get_vector(tokenizer.index_word[index])

        xs = np.zeros((len(mentions), maxlen, 300))
        for i, m in enumerate(mentions):
            before_subject, subject, after_subject = m.anonymous_parts()

            before_seq = tokenizer.texts_to_sequences([before_subject])[0]
            subject_seq = tokenizer.texts_to_sequences([subject])[0]
            after_seq = tokenizer.texts_to_sequences([after_subject])[0]

            before_seq_length = (maxlen - 1 * len(subject_seq)) // 2
            before_seq = before_seq[-before_seq_length:] if len(before_seq) >= before_seq_length else before_seq

            after_seq_length = maxlen - 1 * len(subject_seq) - len(before_seq)
            after_seq = after_seq[:after_seq_length]

            x = before_seq + subject_seq + after_seq
            x = [0] * (maxlen - len(x)) + x  # padding with zeroes
            xs[i] = np.array([index2vector(i) for i in x])
        return xs

    def _mentions_to_ys(self, mentions: typing.List[MentionExpanded]):
        return np_utils.to_categorical([self.__sentiment2int[m.sentiment] for m in mentions], 3)
