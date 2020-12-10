from trainer import api
import typing
from sklearn.model_selection import train_test_split
import os
import pickle
import logging

DATA_DIR = "Data"
BBGH_BACKEND_URL = "http://vps-3c1c3381.vps.ovh.net:8080"


class Dataset(typing.NamedTuple):
    train_x: list
    train_y: list
    val_x: list
    val_y: list
    test_x: list
    test_y: list


class Mentions(object):
    __sentiments = {
        'POSITIVE': 1,
        'NEUTRAL': 0,
        'NEGATIVE': -1
    }

    def __init__(self, mentions: typing.List[api.ApiMentionExpanded]):
        train_mentions, test_mentions = train_test_split(mentions, test_size=0.2, random_state=42)
        train_mentions, val_mentions = train_test_split(train_mentions, test_size=0.2, random_state=42)
        self.train_mentions = train_mentions
        self.val_mentions = val_mentions
        self.test_mentions = test_mentions

    def dataset(self):
        train_x, train_y = self.__mentions_to_xy(self.train_mentions)
        val_x, val_y = self.__mentions_to_xy(self.val_mentions)
        test_x, test_y = self.__mentions_to_xy(self.test_mentions)
        return Dataset(train_x, train_y, val_x, val_y, test_x, test_y)

    def __mentions_to_xy(self, mentions: typing.List[api.ApiMentionExpanded]):
        x = [mention.comment_content for mention in mentions]
        y = [self.__sentiments[mention.sentiment] for mention in mentions]
        return x, y


def get_mentions() -> Mentions:
    file = DATA_DIR + "/mentions.pickle"
    if not os.path.isfile(file):
        mentions = __download_mentions()
        with open(file, 'wb') as handle:
            pickle.dump(mentions, handle)
        return mentions
    with open(file, 'rb') as handle:
        return pickle.load(handle)


def __download_mentions() -> Mentions:
    mentions = list()
    client = api.ApiClient(BBGH_BACKEND_URL)
    for mention in client.all_checked_mentions():
        mentions.append(mention)
    return Mentions(mentions)
