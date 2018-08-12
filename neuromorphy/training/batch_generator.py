# -*- coding: utf-8 -*-

from collections import defaultdict
import math
from typing import Mapping, List, Union

import numpy as np
import tensorflow as tf

from data_info_builder import DataInfoBuilder
from corpus_iterator import CorpusIterator


class BatchGenerator:
    class Generator:
        def __init__(self, generator, max_count):
            self._generator = generator
            self._cur_count = 0
            self._max_count = max_count

        def get_next(self):
            assert self._cur_count < self._max_count
            self._cur_count += 1
            return next(self._generator)

        def has_next(self):
            return self._cur_count < self._max_count

        def left_count(self):
            return self._max_count - self._cur_count

        def reset(self):
            assert self._cur_count == self._max_count
            self._cur_count = 0

        @property
        def batchs_count(self):
            return self._max_count

    def __init__(self, data_info: DataInfoBuilder, corpus_iterator: CorpusIterator, batch_size: int=1024):
        self._data_info = data_info
        self._batch_size = batch_size

        buckets = self._build_buckets(corpus_iterator)
        self._generators = \
            [self._build_data_generator(bucket_data, sent_len) for sent_len, bucket_data in buckets.items()]

    def _build_buckets(self, corpus_iterator: CorpusIterator) -> Mapping[int, List[np.ndarray]]:
        buckets = defaultdict(list)
        with corpus_iterator:
            for sentence in corpus_iterator:
                bucket_size = self._get_bucket_size(len(sentence))

                data = [self._data_info.get_word_index(tok.token) for tok in sentence]
                labels = [self._data_info.get_label_index(tok.grammar_value) for tok in sentence]
                lemma_labels = [self._data_info.get_lemmatize_rule_index(tok.token, tok.lemma) for tok in sentence]
                data = [data, labels, lemma_labels]

                if bucket_size in buckets:
                    for i, feat_data in enumerate(data):
                        buckets[bucket_size][i].append(feat_data)
                else:
                    buckets[bucket_size] = [[feat_data] for feat_data in data]

        return {sent_len: [self._concatenate(data, sent_len) for data in bucket]
                for sent_len, bucket in buckets.items()}

    @staticmethod
    def _get_bucket_size(sent_len: int) -> int:
        bucket_upper_limit = 4
        while bucket_upper_limit < sent_len:
            bucket_upper_limit *= 2
        return bucket_upper_limit

    @staticmethod
    def _concatenate(data: List[Union[List, np.ndarray]], sent_len: int) -> np.ndarray:
        matrix_shape = (sent_len, len(data))
        if isinstance(data[0], np.ndarray):
            matrix_shape += data[0].shape[1:]

        matrix = np.zeros(matrix_shape)
        for i, sent in enumerate(data):
            matrix[:len(sent), i] = sent
        return matrix

    def _build_data_generator(self, data: List[np.ndarray], sent_len: int) -> Generator:
        indices = np.arange(data[0].shape[1])
        batch_size = max(self._batch_size // sent_len, 1)
        batchs_count = int(math.ceil(len(indices) / batch_size))

        def _batch_generator():
            while True:
                np.random.shuffle(indices)
                for i in range(batchs_count):
                    batch_begin = i * batch_size
                    batch_end = min((i + 1) * batch_size, len(indices))
                    batch_indices = indices[batch_begin: batch_end]

                    batch_data = data[0][:, batch_indices]
                    batch_labels = data[1][:, batch_indices]
                    batch_lemma_labels = data[2][:, batch_indices]

                    seq_lengths = (batch_labels == 0).argmax(axis=0)
                    seq_lengths[seq_lengths == 0] = batch_labels.shape[0]
                    max_seq_len = seq_lengths.max()

                    batch_data = batch_data[:max_seq_len]
                    batch_labels, batch_lemma_labels = batch_labels[:max_seq_len], batch_lemma_labels[:max_seq_len]

                    yield batch_data, batch_labels, batch_lemma_labels

        return self.Generator(_batch_generator(), batchs_count)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            while self._has_generators():
                generator = self._sample_generator()
                assert generator.has_next()
                return generator.get_next()
            self._reset_generators()

    def _has_generators(self) -> bool:
        return any(gen.has_next() for gen in self._generators)

    def _sample_generator(self) -> Generator:
        dist = np.array([gen.left_count() for gen in self._generators], dtype=np.float)
        dist /= dist.sum()
        return np.random.choice(self._generators, p=dist)

    def _reset_generators(self):
        for gen in self._generators:
            gen.reset()

    @property
    def batchs_count(self) -> int:
        return sum(gen.batchs_count for gen in self._generators)

    def make_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset() \
                      .from_generator(lambda: self,
                                      output_types=(tf.int64, tf.int64, tf.int64),
                                      output_shapes=(tf.TensorShape([None, None]),
                                                     tf.TensorShape([None, None]),
                                                     tf.TensorShape([None, None])))
