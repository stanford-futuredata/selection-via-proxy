import os
import abc
import gzip
from abc import ABCMeta
from typing import Optional, List, Callable, Any, Mapping, Type

import pandas as pd
import numpy as np
import torch.utils.data as data
from torchvision import transforms

from svp.amazon import transforms as text_transforms

DATASETS = ['amazon_review_full', 'amazon_review_polarity']


class TextClassification(data.Dataset, metaclass=ABCMeta):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.train_data, self.train_labels = self.load_train_data()
            assert self.train_labels.dtype == np.int64
        else:
            self.test_data, self.test_labels = self.load_test_data()
            assert self.test_labels.dtype == np.int64

        self._classes = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (doc, target) where target is index of the target class.
        """
        if self.train:
            doc, target = self.train_data[index], self.train_labels[index]
        else:
            doc, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            doc = self.transform(doc)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return doc, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))  # noqa: E501
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))  # noqa: E501
        return fmt_str

    @abc.abstractmethod
    def load_train_data(self):
        pass

    @abc.abstractmethod
    def load_test_data(self):
        pass

    @property
    def classes(self):
        if self._classes is None:
            self._classes = len(set(self.train_labels)) if self.train else len(set(self.test_labels))  # noqa: E501
        return self._classes


class XiangZhangDataset(TextClassification):
    def load_train_data(self):
        assert self.dirname in self.root
        return self.load_data(os.path.join(self.root, 'train.csv'))

    def load_test_data(self):
        assert self.dirname in self.root
        return self.load_data(os.path.join(self.root, 'test.csv'))

    def load_data(self, path):
        df = pd.read_csv(path, header=None, keep_default_na=False,
                         names=self.columns)
        labels = (df[self.columns[0]] - df[self.columns[0]].min()).values
        if len(self.columns) > 2:
            data = (df[self.columns[1]]
                    .str
                    .cat([df[col] for col in self.columns[2:]], sep=" ")
                    .values)
        else:
            data = df[self.columns[1]].values
        return data, labels

    @property
    @abc.abstractmethod
    def dirname(self):
        pass

    @property
    @abc.abstractmethod
    def columns(self):
        pass


class AmazonProductReviews(XiangZhangDataset):
    dirname = ""
    columns = ['rating', 'subject', 'body']


class AmazonReviewPolarity(AmazonProductReviews):
    dirname = "amazon_review_polarity_csv"


class AmazonReviewFull(AmazonProductReviews):
    dirname = "amazon_review_full_csv"


class AGNews(XiangZhangDataset):
    dirname = "ag_news_csv"
    columns = ['class_index', 'title', 'description']


class DBPedia(XiangZhangDataset):
    dirname = "dbpedia_csv"
    columns = ['class_index', 'title', 'content']


class SogouNews(XiangZhangDataset):
    dirname = "sogou_news_csv"
    columns = ['class_index', 'title', 'content']


class YahooAnswers(XiangZhangDataset):
    dirname = "yahoo_answers_csv"
    columns = ['class_index', 'question_title', 'question_content',
               'best_answer']


class YelpReviewFull(XiangZhangDataset):
    dirname = "yelp_review_full_csv"
    columns = ['rating', 'review']


class YelpReviewPolarity(YelpReviewFull):
    dirname = "yelp_review_polarity_csv"


def parse(path):
    with gzip.open(path, 'rb') as file:
        for line in file:
            yield eval(line)


class GZIPAmazonProductReviews(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.labels = self.load_data()

        self._classes = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (doc, target) where target is index of the target class.
        """
        doc, target = self.data[index], self.labels[index]

        if self.transform is not None:
            doc = self.transform(doc)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return doc, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))  # noqa: E501
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))  # noqa: E501
        return fmt_str

    def load_data(self):
        keep = ('overall', 'summary', 'reviewText')
        records = []
        for raw_record in parse(self.root):
            record = {key: raw_record[key] for key in keep}
            records.append(record)

        df = pd.DataFrame.from_records(records)
        labels = (df['overall'] - df['overall'].min()).astype(np.int64)
        data = (df['summary']
                .str
                .cat([df['reviewText']], sep=" ")
                .values)
        return data, labels

    @property
    def classes(self):
        if self._classes is None:
            self._classes = len(set(self.labels))
        return self._classes


_DATASETS: Mapping[str, Type[XiangZhangDataset]] = {
    'amazon_review_full': AmazonReviewFull,
    'amazon_review_polarity': AmazonReviewPolarity,
    'ag_news': AGNews,
    'dbpedia': DBPedia,
    'sogou_news': SogouNews,
    'yahoo_answers': YahooAnswers,
    'yelp_review_full': YelpReviewFull,
    'yelp_review_polarity': YelpReviewPolarity,
    'amazon': AmazonProductReviews,
}


Transform = Callable[[Any], Any]


def create_dataset(dataset: str, datasets_dir: str,
                   transform: Optional[List[Transform]] = None,
                   target_transform: Optional[List[Transform]] = None,
                   train: bool = True) -> XiangZhangDataset:
    dataset_dir = os.path.join(datasets_dir, dataset + "_csv")
    if transform is not None:
        raw_transforms = transform
    else:
        vocab = text_transforms.Vocab("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/|_#$%ˆ&*~`+=<>()[]{} ",  # noqa: E501
                                      offset=2, unknown=1)
        raw_transforms = transforms.Compose([
            transforms.Lambda(lambda doc: doc.lower()),
            vocab,
            text_transforms.PadOrTruncate(1014),
            transforms.Lambda(lambda doc: doc.astype(np.int64))
        ])
    _dataset = _DATASETS[dataset](root=dataset_dir, train=train,
                                  transform=raw_transforms,
                                  target_transform=target_transform)
    print(_dataset)
    return _dataset
