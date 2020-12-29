import datetime
import time
import typing

import requests
import urllib3.connection


class Error(ConnectionError):
    pass


class Article(typing.NamedTuple):
    id: int
    content: str
    updated_at: datetime.datetime

    @staticmethod
    def from_data(data: dict) -> 'Article':
        if data['updatedAt']:
            updated_at = datetime.datetime.strptime(data['updatedAt'], '%Y-%m-%dT%H:%M:%S')
        else:
            updated_at = datetime.datetime.fromtimestamp(0)
        return Article(
            id=data['id'],
            content=data['content'],
            updated_at=updated_at
        )


class Comment(typing.NamedTuple):
    id: int
    content: str

    @staticmethod
    def from_data(data: dict) -> 'Comment':
        return Comment(
            id=data['id'],
            content=data['content']
        )


class MentionExpanded(object):
    def __init__(self, data: dict):
        self.id = data['id']
        self.article_id = data['articleId']
        self.comment_id = data['commentId']
        self.comment_content = data['commentContent']
        self.player_id = data['playerId']
        self.sentiment = data['mentionSentiment']
        self.starts_at = data['startsAt']
        self.ends_at = data['endsAt']
        self.sentiment_marked_by_human = data['sentimentMarkedByHuman']

    def parts(self):
        before_subject = self.comment_content[0:self.starts_at]
        subject = self.comment_content[self.starts_at:self.ends_at]
        after_subject = self.comment_content[self.ends_at:]
        return before_subject, subject, after_subject

    def anonymous_parts(self):
        before, subject, after = self.parts()
        return before, "przedmiotopinii", after

    def anonymous_comment_content(self):
        before, subject, after = self.anonymous_parts()
        return before + subject + after


class Api(object):
    def __init__(self, host: str):
        self.host = host

    def articles(self, updated_after: datetime.datetime = None) -> dict:
        return self._get_list("/api/articles", {'updatedAfter': self._date_or_none(updated_after)})

    def all_articles(self, updated_after: datetime.datetime = None, sort: str = None) -> typing.Iterator[Article]:
        for item in self._all_items('/api/articles',
                                    {'updatedAfter': self._date_or_none(updated_after),
                                     'sort': sort}):
            yield Article.from_data(item)

    def all_article_comments(self, article_id) -> typing.Iterator[Comment]:
        for item in self._all_items('/api/articles/{}/comments'.format(article_id)):
            yield Comment.from_data(item)

    def all_comments(self, ) -> typing.Iterator[Comment]:
        for article in self.all_articles():
            for comment in self.all_article_comments(article.id):
                yield comment

    def mentions(self) -> dict:
        return self._get_list("/api/mentions", {})

    def all_mentions(self, sort: str = None) -> typing.Iterator[MentionExpanded]:
        params = {'sort': sort}
        for item in self._all_items('/api/mentions', params):
            yield MentionExpanded(item)

    def all_checked_mentions(self, ) -> typing.Iterator[MentionExpanded]:
        for item in self._all_items('/api/mentions', {'sentiment': ['POSITIVE', 'NEUTRAL', 'NEGATIVE']}):
            yield MentionExpanded(item)

    def all_not_checked_mentions(self, ) -> typing.Iterator[MentionExpanded]:
        for item in self._all_items('/api/mentions', {'sentiment': ['NOT_CHECKED']}):
            yield MentionExpanded(item)

    def update_mention_sentiment(self, mention_id: int, sentiment: str):
        return self._post("/api/mentions/{}/sentiment".format(mention_id), {
            "human": False,
            "mentionSentiment": sentiment,
        })

    def update_mentions_sentiments(self, positive_ids, neutral_ids, negative_ids):
        return self._post("/api/mentions/sentiments", {
            "items": [
                {
                    "ids": positive_ids,
                    "sentiment": "POSITIVE"
                },
                {
                    "ids": neutral_ids,
                    "sentiment": "NEUTRAL"
                },
                {
                    "ids": negative_ids,
                    "sentiment": "NEGATIVE"
                }
            ],
        })

    def _post(self, path: str, json: dict):
        resp = requests.post(self.host + path, json=json)
        if resp.status_code not in [200, 201]:
            raise Error('POST {} {} {} {}'.format(path, json, resp.status_code, resp.json()))
        return resp

    def _get_list(self, path: str, params) -> dict:
        try:
            resp = requests.get(self.host + path, params=params)
        except urllib3.connection.HTTPConnection:
            print("HTTP Connection error when doing GET {}. Going to sleep for 3s...".format(path))
            time.sleep(3)
            return self._get_list(path, params)

        if resp.status_code != 200:
            # This means something went wrong.
            raise Error('GET {} {}'.format(path, resp.status_code))
        return resp.json()

    def _all_items(self, path: str, params: typing.Optional[dict] = None) -> typing.Iterator[dict]:
        if params is None:
            params = {}
        items_per_page = 250
        count = items_per_page
        page = 0
        while count == items_per_page:
            items = self._get_list(path, {**params, **{'page': page, 'size': items_per_page}})
            page = page + 1
            count = items["numberOfElements"]
            for item in items['content']:
                yield item

    @staticmethod
    def _date_or_none(date) -> str:
        return date.strftime("%Y-%m-%dT%H:%M:%S") if date is not None else None
