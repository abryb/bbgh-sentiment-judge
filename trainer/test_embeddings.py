import gensim
from resource import getrusage, RUSAGE_SELF
import time
import pickle
from gensim.models import Word2Vec

if __name__ == '__main__':
    # with open('Data/word_embeddings.pickle', 'rb') as handle:
    #     word2vec_model = pickle.load(handle)
    # time.sleep(60)
    # print(getrusage(RUSAGE_SELF).ru_maxrss)

    model = Word2Vec.load("Data/word2vec.model")

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("Data/nkjp+wiki-forms-all-300-cbow-hs-50.txt",binary=False)
    # time.sleep(5)
    word2vec_model.save("Data/word2vec.model")
    # print(getrusage(RUSAGE_SELF).ru_maxrss)
    # time.sleep(5)
    # print(getrusage(RUSAGE_SELF).ru_maxrss)
    # with open('Data/word_embeddings.pickle', 'wb') as handle:
    #     pickle.dump(word2vec_model, handle)
    # print(getrusage(RUSAGE_SELF).ru_maxrss)
