import pickle
from gensim.models.keyedvectors import Word2VecKeyedVectors

DATA_DIR = 'Data'


def get_word2vec() -> Word2VecKeyedVectors:
    with open(DATA_DIR+"/word2vec.pickle", 'rb') as handle:
        return pickle.load(handle)