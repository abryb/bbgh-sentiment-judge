import os

from sklearn.model_selection import train_test_split
from tabulate import tabulate

import trainer.api
import trainer.cache
import trainer.model
import trainer.repository
import trainer.word2vector


root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class Worker(object):
    def __init__(self,
                 backend_host=None,
                 cache_dir=None,
                 models_dir=None
                 ):
        backend_host = backend_host or "http://vps-3c1c3381.vps.ovh.net:8080"
        cache_dir = cache_dir or root_dir + "/Data"
        models_dir = models_dir or root_dir + "/Models"
        cache_dir = os.path.realpath(cache_dir)
        models_dir = os.path.realpath(models_dir + "/default")

        self.cache = trainer.cache.FileCache(cache_dir)
        self.api = trainer.api.Api(backend_host)
        self.repository = trainer.repository.Repository(self.cache, self.api)
        self.word2vector = trainer.word2vector.Word2Vector(self.cache, self.repository)
        self.model = trainer.model.Model(self.word2vector)
        self.models_dir = models_dir
        self.model_loaded = False

    def prepare_word2vec(self):
        self.repository.download_articles_and_comments()
        self.word2vector.train_on_articles_and_comments()

    def download_mentions(self):
        self.repository.download_mentions(only_missing=False)
        self.word2vector.train_on_mentions()
        self._split_mentions()

    def train(
            self,
            epochs=None,
            batch_size=None,
            maxlen=None,
            val=False,
            save=False,
            train_on_all=False,
            verbose=1
    ):
        epochs = epochs or 44
        batch_size = batch_size or 32
        maxlen = maxlen or 32

        train_mentions, val_mentions, test_mentions = self._get_mentions()

        if train_on_all:
            train_mentions = train_mentions + test_mentions + val_mentions
            test_mentions = None
            val_mentions = None

        if not val and not train_on_all:
            train_mentions, val_mentions = train_mentions + val_mentions, None

        self.model.train(train_mentions,
                         val_mentions=val_mentions,
                         epochs=epochs,
                         batch_size=batch_size,
                         maxlen=maxlen,
                         verbose=verbose
                         )

        if test_mentions is not None:
            self.model.evaluate(test_mentions)
        if save:
            print("Saving model.")
            self.model.save(self.models_dir)
        self.model_loaded = True

    def evaluate(self):
        self._load_model()
        _, _, test_mentions = self._get_mentions()
        self.model.evaluate(test_mentions)

    def predict(self, only_not_checked=True):
        self._load_model()
        if only_not_checked:
            mentions = self.repository.get_mentions_without_predictions()
        else:
            mentions = self.repository.get_mentions_not_marked_by_human()
        mentions = list(mentions)
        if len(mentions) == 0:
            print("Nothing to predict.")
            return
        print("Running predict for {} mentions".format(len(mentions)))

        sentiments = self.model.predict(mentions)

        predictions = list()
        for i, m in enumerate(mentions):
            predictions.append(trainer.repository.Prediction(m.id, sentiments[i]))

        self.repository.save_predictions(predictions)

    def publish(self, only_unpublished=True):
        if only_unpublished:
            predictions = self.repository.get_unpublished_predictions()
        else:
            predictions = self.repository.get_predictions()

        marked_by_human_ids = [m.id for m in self.repository.get_mentions() if m.sentiment_marked_by_human]
        predictions = [p for p in predictions if p.mention_id not in marked_by_human_ids]

        positive_ids = [p.mention_id for p in predictions if p.sentiment == 'POSITIVE']
        neutral_ids = [p.mention_id for p in predictions if p.sentiment == 'NEUTRAL']
        negative_ids = [p.mention_id for p in predictions if p.sentiment == 'NEGATIVE']

        print("Publishing {} predictions".format(len(predictions)))
        self.api.update_mentions_sentiments(positive_ids, neutral_ids, negative_ids)

        for p in predictions:
            p.published = True
        self.repository.save_predictions(predictions)

    def show_prediction(self, mention_id):
        prediction = self.repository.get_prediction(mention_id)
        if prediction is None:
            print("There is no prediction for this mention.")
        else:
            print(tabulate([
                [mention_id, prediction.sentiment, prediction.published],
            ], headers=['Mention ID', 'Sentiment', 'Published']))

    def run(self, train=False):
        if train:
            self.download_mentions()
            if train:
                self.train(
                    train_on_all=True,
                    save=True,
                    verbose=2
                )

            self.predict(only_not_checked=False)
            self.publish(only_unpublished=False)
        else:
            mentions = self.repository.download_not_checked_mentions()
            if len(mentions) > 0:
                self.predict(only_not_checked=True)
                self.publish(only_unpublished=True)
            else:
                print("0 mentions found. Nothing to do here...")

    def _get_mentions(self):
        return self.cache.get_or_create("worker.dataset", self._split_mentions)

    def _load_model(self):
        if not self.model_loaded:
            self.model.load(self.models_dir)
            self.model_loaded = True

    def _split_mentions(self):
        mentions = list(self.repository.get_checked_mentions_marked_by_human())
        train_mentions, test_mentions = train_test_split(mentions, test_size=0.2)
        train_mentions, val_mentions = train_test_split(train_mentions, test_size=0.2)
        self.cache.save("worker.dataset", (train_mentions, val_mentions, test_mentions))
        return train_mentions, val_mentions, test_mentions

