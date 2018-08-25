# -*- coding: utf-8 -*-

import os
import pickle
from collections import Counter
from string import punctuation
from os.path import commonprefix
from typing import Tuple, Set, Mapping, List

import numpy as np
import attr

from neuromorphy.dictionary import MorphoAnalyser
from neuromorphy.train.corpus_iterator import CorpusIterator


@attr.s(frozen=True)
class DataInfo:
    _morph: MorphoAnalyser = attr.ib()
    _lemmatize_rule_index: Mapping[Tuple[int, str], int] = attr.ib()
    _label_index: Mapping[str, int] = attr.ib()
    _char_index: Mapping[str, int] = attr.ib()
    _word_index: Mapping[str, int] = attr.ib()
    _max_word_len: int = attr.ib()
    _chars_matrix: np.ndarray = attr.ib()
    _grammemes_matrix: np.ndarray = attr.ib()

    def get_char_index(self, char: str) -> int:
        return self._char_index[char] if char in self._char_index else DataInfoBuilder.UNK_SYMB_INDEX

    def get_word_index(self, word: str) -> int:
        return self._word_index[word] if word in self._word_index else DataInfoBuilder.EMPTY_SYMB_INDEX

    def get_label_index(self, label: str) -> int:
        return self._label_index[label] if label in self._label_index else DataInfoBuilder.UNK_SYMB_INDEX

    def get_lemmatize_rule_index(self, word: str, lemma: str) -> int:
        # +1 for padding (we don't want to process rules which did not appear in the Morphology)
        return self._get_lemmatize_rule(word, lemma) + 1

    def _get_lemmatize_rule(self, word: str, lemma: str) -> int:
        def predict_lemmatize_rule(word: str, lemma: str) -> Tuple[int, str]:
            if len(word) == 0:
                return 0, lemma

            common_prefix = commonprefix([word, lemma])
            if len(common_prefix) == 0:
                cut, append = predict_lemmatize_rule(word[1:], lemma)
                return cut + 1, append

            return len(word) - len(common_prefix), lemma[len(common_prefix):]

        parses = self._morph.analyse_word(word)
        lemmas = set((parse.lemma, parse.lemmatize_rule_index) for parse in parses)
        if len(lemmas) == 1:
            return parses[0].lemmatize_rule_index
        for parsed_lemma, rule_index in lemmas:
            parsed_lemma = parsed_lemma.lower().replace('ё', 'е')
            if parsed_lemma.endswith(lemma) \
                    or (lemma.endswith('ся') and parsed_lemma.endswith(lemma[:-2])) \
                    or (parsed_lemma.endswith('ся') and parsed_lemma[:-2].endswith(lemma)):
                return rule_index

        word = word.lower().replace('ё', 'е')
        restored_rule = predict_lemmatize_rule(word, lemma)
        if restored_rule in self._lemmatize_rule_index:
            return self._lemmatize_rule_index[restored_rule]
        else:
            return -1

    @property
    def chars_count(self) -> int:
        return len(self._char_index)

    @property
    def labels_count(self) -> int:
        return len(self._label_index)

    @property
    def lemma_labels_count(self) -> int:
        return len(self._morph.lemmatize_rule_mapping) + 1

    @property
    def chars_matrix(self) -> np.ndarray:
        return self._chars_matrix

    @property
    def grammemes_matrix(self) -> np.ndarray:
        return self._grammemes_matrix

    @property
    def word_index(self) -> Mapping:
        return self._word_index

    # TODO: lazy
    @property
    def labels(self) -> Tuple[str]:
        return tuple(label for label, _ in sorted(self._label_index.items(), key=lambda x: x[1]))

    def save(self, path: str):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'data_info.pkl'), 'wb') as f:
            pickle.dump((
                self._lemmatize_rule_index, self._label_index, self._char_index,
                self._word_index, self._max_word_len
            ), file=f, protocol=pickle.HIGHEST_PROTOCOL)
        np.savez(os.path.join(path, 'embeddings.npz'),
                 grammemes_matrix=self._grammemes_matrix, chars_matrix=self._chars_matrix)

    @classmethod
    def restore(cls, morph: MorphoAnalyser, path: str):
        with open(os.path.join(path, 'data_info.pkl'), 'rb') as f:
            (lemmatize_rule_index, label_index, char_index,
                word_index, max_word_len) = pickle.load(f)

        embeddings = np.load(os.path.join(path, 'embeddings.npz'))
        chars_matrix = embeddings['chars_matrix']
        grammemes_matrix = embeddings['grammemes_matrix']

        return cls(morph=morph,
                   lemmatize_rule_index=lemmatize_rule_index,
                   label_index=label_index,
                   char_index=char_index,
                   word_index=word_index,
                   max_word_len=max_word_len,
                   chars_matrix=chars_matrix,
                   grammemes_matrix=grammemes_matrix)


class DataInfoBuilder:
    EMPTY_SYMB = ''
    EMPTY_SYMB_INDEX = 0
    UNK_SYMB = '<unk>'
    UNK_SYMB_INDEX = 1

    def __init__(self, morph: MorphoAnalyser, train_path: str, val_path: str=None, test_path: str=None):
        self._morph = morph
        self._lemmatize_rule_index = {rule: index for index, rule in enumerate(morph.lemmatize_rule_mapping)}

        self._label_index, self._char_index, self._max_word_len = self._collect_mappings(train_path)

        word_index = self._initial_mapping_values(add_unk=False)
        chars_matrix = [[]]
        grammemes_matrix = [np.zeros(self._morph.grammemes_count)]

        self._build_embeddings(train_path, word_index, chars_matrix, grammemes_matrix)
        if val_path is not None:
            self._build_embeddings(val_path, word_index, chars_matrix, grammemes_matrix)
        if test_path is not None:
            self._build_embeddings(test_path, word_index, chars_matrix, grammemes_matrix)

        self._word_index = word_index
        self._grammemes_matrix, self._chars_matrix = self._convert_embeddings(grammemes_matrix, chars_matrix)

    @staticmethod
    def _initial_mapping_values(add_unk: bool=True):
        default_mapping = {DataInfoBuilder.EMPTY_SYMB: DataInfoBuilder.EMPTY_SYMB_INDEX}
        if add_unk:
            default_mapping[DataInfoBuilder.UNK_SYMB] = DataInfoBuilder.UNK_SYMB_INDEX
        return default_mapping

    def _collect_mappings(self, path: str) -> Tuple[Mapping[str, int], Mapping[str, int], int]:
        label_index = self._initial_mapping_values(add_unk=True)

        chars_counter = Counter()
        lengths_counter = Counter()

        with CorpusIterator(path) as corpus_iterator:
            for sentence in corpus_iterator:
                for token in sentence:
                    if token.grammar_value not in label_index:
                        label_index[token.grammar_value] = len(label_index)

                    lengths_counter[len(token.token)] += 1

                    for char in token.token:
                        chars_counter[char] += 1

        chars = set(char for char, _ in chars_counter.most_common(self._find_count(chars_counter, 0.99)))
        chars |= self._get_range('0', '9') | set(punctuation)
        char_index = self._initial_mapping_values(add_unk=True)
        for char in chars:
            char_index[char] = len(char_index)

        max_word_len = self._find_max_len(lengths_counter, 0.98)

        return label_index, char_index, max_word_len

    @staticmethod
    def _find_count(counter: Counter, threshold: float) -> int:
        cur_sum_count = 0
        sum_count = sum(counter.values())
        for i, (_, count) in enumerate(counter.most_common()):
            cur_sum_count += count
            if cur_sum_count > sum_count * threshold:
                return i
        return len(counter)

    @staticmethod
    def _find_max_len(counter: Counter, threshold: float) -> int:
        sum_count = sum(counter.values())
        cum_count = 0
        for i in range(max(counter)):
            cum_count += counter[i]
            if cum_count > sum_count * threshold:
                return i
        return max(counter)

    @staticmethod
    def _get_range(first_symb: str, last_symb: str) -> Set:
        return set(chr(c) for c in range(ord(first_symb), ord(last_symb) + 1))

    def _build_embeddings(self, path: str, word_index: Mapping[str, int],
                          chars_matrix: List[List[int]], grammemes_matrix: List[np.ndarray]):
        def _get_char_index(char: str) -> int:
            return self._char_index[char] if char in self._char_index else DataInfoBuilder.UNK_SYMB_INDEX

        with CorpusIterator(path) as corpus_iterator:
            for sentence in corpus_iterator:
                for token in sentence:
                    if token.token not in word_index:
                        word_index[token.token] = len(word_index)

                        chars_vector = [_get_char_index(char) for char in token.token[-self._max_word_len:]]
                        chars_matrix.append(chars_vector)

                        grammemes_vector = self._morph.build_grammemes_vector(token.token)
                        grammemes_matrix.append(grammemes_vector)

    def _convert_embeddings(self, grammemes_matrix: List[np.ndarray], chars_matrix: List[List[int]])  \
            -> Tuple[np.ndarray, np.ndarray]:
        grammemes_matrix = np.array(grammemes_matrix)

        new_chars_matrix = np.zeros((len(chars_matrix), self._max_word_len))
        for i, chars in enumerate(chars_matrix):
            if len(chars) == 0:
                continue
            new_chars_matrix[i, -len(chars):] = chars
        return grammemes_matrix, new_chars_matrix

    def build(self) -> DataInfo:
        return DataInfo(morph=self._morph,
                        lemmatize_rule_index=self._lemmatize_rule_index,
                        label_index=self._label_index,
                        char_index=self._char_index,
                        word_index=self._word_index,
                        max_word_len=self._max_word_len,
                        chars_matrix=self._chars_matrix,
                        grammemes_matrix=self._grammemes_matrix)

def main():
    from tempfile import TemporaryDirectory

    morph = MorphoAnalyser('RussianDict.zip')
    data_info = DataInfoBuilder(morph=morph,
                                train_path='UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu',
                                val_path='UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu',
                                test_path='UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu').build()

    print(data_info.get_char_index('у'))
    print(data_info.get_word_index('селу'))
    print(morph.lemmatize_rule_mapping[data_info.get_lemmatize_rule_index('селу', 'село') - 1])

    with TemporaryDirectory() as tmpdir:
        data_info.save(tmpdir)
        data_info_restored = DataInfo.restore(morph, tmpdir)

        print(data_info_restored.get_char_index('у'))
        print(data_info_restored.get_word_index('селу'))
        print(morph.lemmatize_rule_mapping[data_info_restored.get_lemmatize_rule_index('селу', 'село') - 1])


if __name__ == '__main__':
    main()
