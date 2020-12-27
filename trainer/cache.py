import os
import pickle
from pathlib import Path


class FileCache(object):
    def __init__(self, directory: str):
        self.directory = directory
        self.values = {}

    def save(self, name, variable):
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        self.values[name] = variable
        with open(self._filename(name), 'wb') as handle:
            pickle.dump(variable, handle)

    def has(self, name):
        return os.path.exists(self._filename(name))

    def get(self, name):
        if name not in self.values:
            with open(self._filename(name), 'rb') as handle:
                value = pickle.load(handle)
                self.values[name] = value
        return self.values[name]

    def get_or_create(self, name, create):
        if self.has(name):
            return self.get(name)
        else:
            result = create()
            if not self.has(name):  # save if create function didn't saved it already
                self.save(name, result)
            return result

    def _filename(self, name):
        return os.path.join(self.directory, name + ".pickle")
