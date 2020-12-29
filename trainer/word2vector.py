import os
import tempfile
import zipfile

import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer

import trainer.cache
import trainer.repository
from trainer.utils import download_url


class Word2Vector(object):
    neutral_vector = np.repeat(1, 300)
    zero_vector = np.zeros(300)

    def __init__(self, cache: trainer.cache.FileCache, repository: trainer.repository.Repository):
        self.cache = cache
        self.repository = repository

    def get_vector(self, word: str):
        m = self._get_model_trained_on_mentions()
        if word in m.wv.vocab:
            return m.wv.get_vector(word)
        return self.neutral_vector

    def train_on_articles_and_comments(self):
        print("Training word2vec model on articles and comments.")
        model = self._get_model_pretrained()
        articles, comments = self.repository.get_articles_and_comments()
        texts = [x.content for x in articles] + [x.content for x in comments]
        self._train_model(model, texts)
        self.cache.save('word2vector.model.trained_on_articles_and_comments', model)
        return model

    def train_on_mentions(self):
        print("Training word2vec model on mentions.")
        model = self._get_model_trained_on_articles_and_comments()
        mentions = self.repository.get_mentions()
        texts = [m.anonymous_comment_content() for m in mentions]
        self._train_model(model, texts)
        self.cache.save('word2vector.model.trained_on_mentions', model)
        return model

    def _get_model_trained_on_mentions(self):
        return self.cache.get_or_create('word2vector.model.trained_on_mentions', self.train_on_mentions)

    def _get_model_trained_on_articles_and_comments(self) -> Word2Vec:
        return self.cache.get_or_create(
            'word2vector.model.trained_on_articles_and_comments',
            self.train_on_articles_and_comments)

    def _get_model_pretrained(self) -> Word2Vec:
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

        return self.cache.get_or_create('word2vector.model.pretrained', create)

    def _train_model(self, model: Word2Vec, texts):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        texts_seq = tokenizer.sequences_to_texts(tokenizer.texts_to_sequences(texts))
        texts_seq = [f.split(" ") for f in texts_seq]
        print("Adding to word2vec vocabulary...")
        model.min_count = 2
        model.build_vocab(texts_seq, update=True)
        print("Training word2vec ...")
        model.train(
            texts_seq,
            total_examples=len(texts_seq),
            epochs=model.epochs)
