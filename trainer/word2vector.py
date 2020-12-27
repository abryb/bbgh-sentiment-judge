import os
import tempfile
import typing
import zipfile

import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer

import trainer.cache
import trainer.repository
from trainer.utils import download_url


class Word2Vector(object):
    default_vector = np.zeros(300)

    def __init__(self, cache: trainer.cache.FileCache, repository: trainer.repository.Repository):
        self.cache = cache
        self.repository = repository
        self._word2vector: typing.Optional[dict] = None

    def get_vector(self, word: str):
        if self._word2vector is None:
            self._word2vector = self.cache.get_or_create("word2vector.dictionary",
                                                         self.create_word2vector_dictionary_for_repository_mentions)
        if word in self._word2vector:
            return self._word2vector[word]
        return self.default_vector

    def create_word2vector_dictionary_for_repository_mentions(self):
        print("Creating word2vector dictionary for words in mentions...")
        self._word2vector = self.cache.get_or_create("word2vector.dictionary", lambda: dict())
        mentions = self.repository.get_mentions()
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([m.anonymous_comment_content() for m in mentions])
        missing = list("subject")
        for word in list(tokenizer.word_index):
            if word not in self._word2vector:
                missing.append(word)
        print("Found {} missing words in word2vector dictionary.".format(len(missing)))
        if len(missing) > 0:
            word2vec_model = self._get_trained_on_mentions_model()
            for word in missing:
                self._word2vector[word] = self.default_vector
                if word in word2vec_model.wv.vocab:
                    index = word2vec_model.wv.vocab[word].index
                    self._word2vector[word] = word2vec_model.wv.vectors[index]
            self.cache.save("word2vector.dictionary", self._word2vector)
        return self._word2vector

    def _get_trained_on_mentions_model(self):
        return self.cache.get_or_create('word2vector.trained_on_mentions_model', self.train_word2vec_model_on_mentions)

    def _get_trained_model(self) -> Word2Vec:
        return self.cache.get_or_create('word2vector.trained_model', self.train_word2vec_model)

    def train_word2vec_model_on_mentions(self):
        print("Training word2vec model on mentions.")
        model = self._get_trained_model()
        mentions = self.repository.get_mentions()
        texts = [m.anonymous_comment_content() for m in mentions]
        tokenizer = Tokenizer()
        texts_seq = tokenizer.sequences_to_texts(tokenizer.texts_to_sequences(texts))
        print("Adding to word2vec vocabulary...")
        model.build_vocab(texts_seq, update=True)
        print("Training word2vec ...")
        model.train(
            texts_seq,
            total_examples=len(texts_seq),
            epochs=model.epochs)

        self.cache.save('word2vector.trained_on_mentions_model', model)
        return model

    def train_word2vec_model(self):
        print("Training word2vec model on articles and comments.")
        model = self._get_pretrained_model()
        articles, comments = self.repository.get_articles_and_comments()
        texts = list()
        texts = texts + [x.content for x in articles] + [x.content for x in comments]

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        model.min_count = 2
        texts_seq = tokenizer.sequences_to_texts(tokenizer.texts_to_sequences(texts))
        print("Adding to word2vec vocabulary...")
        model.build_vocab(texts_seq, update=True)
        print("Training word2vec ...")
        model.train(
            texts_seq,
            total_examples=len(texts_seq),
            epochs=model.epochs)

        self.cache.save('word2vector.trained_model', model)
        return model

    def _get_pretrained_model(self) -> Word2Vec:
        def create():
            directory = tempfile.gettempdir()
            pre_trained_model_file = directory + "/nkjp+wiki-forms-all-300-skipg-hs-50"
            if not os.path.exists(pre_trained_model_file):
                zip_file = pre_trained_model_file + ".zip"
                url = "http://dsmodels.nlp.ipipan.waw.pl/binmodels/nkjp+wiki-forms-all-300-skipg-hs-50.zip"
                print("Downloading pretrained word2vec model from {} ...".format(url))
                download_url(url, zip_file)
                print("Extracting zip model file...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(directory)
            model = Word2Vec.load(pre_trained_model_file)

            print("Fixing casing in pretrained model...")
            for word in list(model.wv.vocab):
                if word.lower() != word:
                    if not word.lower() in model.wv.vocab:
                        model.wv.vocab[word.lower()] = model.wv.vocab[word]
                        index = model.wv.vocab[word].index
                        del model.wv.vocab[word]
                        model.wv.index2word[index] = word.lower()
                        model.wv.index2entity[index] = word.lower()

            return model

        return self.cache.get_or_create('word2vector.pre_trained_model', create)
