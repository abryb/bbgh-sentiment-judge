import typing as t
from tqdm import tqdm

import trainer.api as api
import trainer.cache


class Prediction(object):
    def __init__(self, mention_id: int, sentiment: str):
        self.mention_id = mention_id
        self.sentiment = sentiment
        self.published = False


class Repository(object):
    def __init__(self, cache: trainer.cache.FileCache, api: api.Api):
        self.cache = cache
        self.api = api
        self._articles = None
        self._comments = None
        self._mentions = None
        self._predictions = None

    # articles and comments
    def _load_articles_and_comments(self):
        if self._articles is None or self._comments is None:
            self._articles = self.cache.get('repository.articles')
            self._comments = self.cache.get('repository.comments')

    def download_articles_and_comments(self):
        self._articles = {}
        self._comments = {}
        print("Downloading articles and comments...")
        pbar = tqdm(total=self.api.articles()['totalElements'])
        for article in self.api.all_articles():
            self._articles[article.id] = article
            for comment in self.api.all_article_comments(article.id):
                self._comments[comment.id] = comment
            pbar.update(1)
        pbar.close()
        self.cache.save('repository.articles', self._articles)
        self.cache.save('repository.comments', self._comments)

    def get_articles_and_comments(self) -> t.Tuple[t.Iterator[api.Article], t.Iterator[api.Comment]]:
        self._load_articles_and_comments()
        return list(self._articles.values()), list(self._comments.values())

    # Mentions
    def _load_mentions(self):
        if self._mentions is None:
            self._mentions = self.cache.get_or_create("repository.mentions", lambda: dict())

    def download_not_checked_mentions(self) -> t.List[api.MentionExpanded]:
        print("Downloading not checked mentions...")
        mentions = list(self.api.all_not_checked_mentions())
        self.save_mentions(mentions)
        return mentions

    def download_mentions(self, only_missing=True) -> t.List[api.MentionExpanded]:
        self._load_mentions()
        downloaded = list()
        if not only_missing:
            print("Downloading all mentions...")
            self._mentions = {}
        else:
            print("Downloading missing mentions...")
        current_count = len(self._mentions)
        api_count = self.api.mentions()['totalElements']
        to_download = api_count - current_count
        pbar = tqdm(total=to_download)
        for m in self.api.all_mentions(sort='id,DESC'):
            if m.id not in self._mentions:
                downloaded.append(m)
                self._mentions[m.id] = m
                pbar.update(1)
                to_download -= 1
            if to_download == 0:
                break
        pbar.close()
        self.save_mentions()
        return downloaded

    def get_mentions(self) -> t.Iterator[api.MentionExpanded]:
        self._load_mentions()
        return list(self._mentions.values())

    def get_checked_mentions_marked_by_human(self) -> t.Iterator[api.MentionExpanded]:
        marked = filter(lambda x: x.sentiment_marked_by_human in [True], self.get_mentions())
        return filter(lambda x: x.sentiment != 'NOT_CHECKED', marked)

    def get_mentions_not_marked_by_human(self) -> t.Iterator[api.MentionExpanded]:
        return filter(lambda x: x.sentiment_marked_by_human in [False, None], self.get_mentions())

    def get_mentions_without_predictions(self) -> t.Iterator[api.MentionExpanded]:
        self._load_predictions()
        return filter(lambda x: x.id not in self._predictions, list(self.get_mentions_not_marked_by_human()))

    def save_mentions(self, mentions: t.List[api.MentionExpanded] = None):
        mentions = mentions or list()
        self._load_mentions()
        for m in mentions:
            if m.sentiment_marked_by_human and m.sentiment == 'NOT_CHECKED':
                m.sentiment_marked_by_human = False
            self._mentions[m.id] = m
        self.cache.save('repository.mentions', self._mentions)

    # Predictions
    def _load_predictions(self):
        if self._predictions is None:
            self._predictions = self.cache.get_or_create("repository.predictions", lambda: dict())

    def get_predictions(self) -> t.List[Prediction]:
        self._load_predictions()
        return list(self._predictions.values())

    def get_prediction(self, mention_id) -> t.Optional[Prediction]:
        self._load_predictions()
        if mention_id in self._predictions:
            return self._predictions[mention_id]
        return None

    def save_predictions(self, predictions: t.List[Prediction] = None):
        self._load_predictions()
        predictions = predictions or list()
        for p in predictions:
            self._predictions[p.mention_id] = p
        self.cache.save('repository.predictions', self._predictions)

    def get_unpublished_predictions(self) -> t.List[Prediction]:
        return list(filter(lambda x: x.published is False, self.get_predictions()))
