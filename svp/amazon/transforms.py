import numpy as np


class Vocab:
    def __init__(self, tokens=None, offset=0, unknown=None):
        self.mapping = {}
        self.reverse_mapping = {}
        self.offset = offset
        self.unknown = unknown
        for token in tokens:
            self.add_token(token)

    def __len__(self):
        return len(self.mapping) + self.offset

    def __call__(self, doc):
        return self.map_sequence(doc)

    def add_token(self, token):
        if token not in self.mapping:
            index = len(self)
            self.mapping[token] = index
            self.reverse_mapping[index] = token

    def __repr__(self):
        fmt_str = self.__class__.__name__
        fmt_str += "(vocab={0}, offset={1}, unknown={2})".format(
            self.__len__(), self.offset, self.unknown)
        return fmt_str

    def map(self, token, unknown=None):
        if token in self.mapping:
            return self.mapping[token]
        else:
            return unknown if unknown is not None else self.unknown

    def map_sequence(self, tokens, unknown=None):
        return np.array([self.map(token, unknown=unknown) for token in tokens])

    def reverse_map(self, index, unknown=None):
        if index in self.reverse_mapping:
            return self.reverse_mapping[index]
        else:
            return unknown if unknown is not None else self.unknown

    def reverse_map_sequence(self, indices, unknown):
        return [self.reverse_map(index, unknown=unknown) for index in indices]


class PadOrTruncate:
    def __init__(self, max_length, fill=0):
        self.max_length = max_length
        self.fill = fill

    def __call__(self, doc):
        current = len(doc)
        trimmed = doc[:self.max_length]
        padding = [self.fill] * (self.max_length - current)
        return np.concatenate([trimmed, padding])

    def __repr__(self):
        fmt_str = self.__class__.__name__
        fmt_str += "(max_length={0}, fill={1})".format(
            self.max_length, self.fill)
        return fmt_str
