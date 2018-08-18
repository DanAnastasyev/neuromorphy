# -*- coding: utf-8 -*-

import os
from zipfile import ZipFile
from tempfile import TemporaryDirectory
from typing import Tuple, List, Callable, Sequence
import pickle

import dawg
import numpy as np


def _lazy_property(fn):
    '''Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


class Parse:
    def __init__(self, grammemes_mappings: Tuple[Tuple[str]], grammar_value_mapping: Tuple[Tuple[int]],
                 lemmatize_rule_mapping: Tuple[Tuple[int, str]], word: str,
                 grammar_value_index: int, lemmatize_rule_index: int, freq: float):
        self._grammemes_mappings = grammemes_mappings
        self._grammar_value_mapping = grammar_value_mapping
        self._lemmatize_rule_mapping = lemmatize_rule_mapping

        self._grammar_value_index = grammar_value_index

        self.lemmatize_rule_index = lemmatize_rule_index
        self.word = word
        self.frequency = freq

    @_lazy_property
    def lemma(self):
        cut, append = self._lemmatize_rule_mapping[self.lemmatize_rule_index]
        return (self.word[:-cut] if cut != 0 else self.word) + append

    @_lazy_property
    def grammar_value(self):
        return GrammarValue(self._grammemes_mappings, self._grammar_value_mapping,
                            self._grammar_value_index)

    def __str__(self):
        return 'Word - {}; GrammarValue - {}; Frequency - {}'.format(self.word, self.grammar_value, self.frequency)

    def __repr__(self):
        return 'Parse: {}'.format(self)


class GrammarValue:
    def __init__(self, grammemes_mappings: Tuple[Tuple[str]], grammar_value_mapping: Tuple[Tuple[int]],
                 grammar_value_index: int):
        self._grammemes_mappings = grammemes_mappings
        self._grammar_value_mapping = grammar_value_mapping
        self._grammar_value_index = grammar_value_index

    def __str__(self):
        return ' '.join(self._grammemes_mappings[i][grammeme_index]
                        for i, grammeme_index in enumerate(self._grammar_value_mapping[self._grammar_value_index])
                        if grammeme_index != 0)

    def __repr__(self):
        return 'GrammarValue: {}'.format(self)


class MorphoAnalyser:
    def __init__(self, dict_path: str):
        with TemporaryDirectory('dict') as temp_dir:
            with ZipFile(dict_path) as zip_file:
                zip_file.extractall(temp_dir)

            self._dawg = dawg.RecordDAWG('>HHH')
            self._dawg.load(os.path.join(temp_dir, 'dict.dawg'))

            with open(os.path.join(temp_dir, 'dict.info'), 'rb') as f:
                self._categories, self._grammemes_mappings, self._grammar_value_mapping, self._lemmatize_rule_mapping, \
                self._alphabet, self._similar_letters, self._quantized_freqs_mapping = pickle.load(f)

        self._similar_letters_replacements = self._compile_replacements()
        self._grammemes_matrix = self._build_grammemes_matrix()

    def _compile_replacements(self):
        similar_letters_replacements = {}
        for first_letter, second_letter in self._similar_letters:
            similar_letters_replacements[first_letter] = second_letter
            similar_letters_replacements[first_letter.upper()] = second_letter.upper()
        return self._dawg.compile_replaces(similar_letters_replacements)

    def _build_grammemes_matrix(self) -> np.ndarray:
        grammemes_vector_len = sum(len(grammar_category) for grammar_category in self._grammemes_mappings)

        grammemes_matrix = np.zeros((len(self._grammar_value_mapping), grammemes_vector_len))
        for i, grammar_value in enumerate(self._grammar_value_mapping):
            shift = 0
            for grammar_category_index, grammeme in enumerate(grammar_value):
                grammemes_matrix[i, shift + grammeme] = 1.
                shift += len(self._grammemes_mappings[grammar_category_index])
        return grammemes_matrix

    def build_grammemes_vector(self, word: str, is_case_sensitive: bool=False) -> np.ndarray:
        def _build_grammemes_vector(word, grammar_val_index, lemmatize_rule_index, freq):
            freq = max(freq, 1e-9)
            cur_vector = freq * self._grammemes_matrix[grammar_val_index]

            _build_grammemes_vector.grammemes_vector += cur_vector
            _build_grammemes_vector.sum_freq += freq

        _build_grammemes_vector.grammemes_vector = np.zeros(self._grammemes_matrix.shape[1])
        _build_grammemes_vector.sum_freq = 0.

        self._analyse_word(word, is_case_sensitive, _build_grammemes_vector)

        if _build_grammemes_vector.sum_freq != 0.:
            return _build_grammemes_vector.grammemes_vector / _build_grammemes_vector.sum_freq
        else:
            assert np.all(_build_grammemes_vector.grammemes_vector == 0.)
            return _build_grammemes_vector.grammemes_vector

    def analyse_word(self, word: str, is_case_sensitive: bool=False) -> List[Parse]:
        def _collect_parses(word, grammar_val_index, lemmatize_rule_index, freq):
            res.append(self._get_parse(word, grammar_val_index, lemmatize_rule_index, freq))

        res = []
        self._analyse_word(word, is_case_sensitive, _collect_parses)
        return res

    def _analyse_word(self, word: str, is_case_sensitive: bool, callback: Callable[[str, int, int, float], None]):
        if not is_case_sensitive:
            word = word.lower()
        self._analyse_single_word(word, callback)
        if not is_case_sensitive:
            self._analyse_single_word(word.capitalize(), callback)
            self._analyse_single_word(word.upper(), callback)

    def _analyse_single_word(self, word: str, callback: Callable[[str, int, int, float], None]):
        for corrected_word, values in self._dawg.similar_items(word, self._similar_letters_replacements):
            for freq_index, lemmatize_rule_index, grammar_val_index in values:
                callback(corrected_word, grammar_val_index, lemmatize_rule_index,
                         self._quantized_freqs_mapping[freq_index])

    def _get_parse(self, word: str, grammar_val_index: int, lemmatize_rule_index: int, freq: float) -> Parse:
        return Parse(self._grammemes_mappings, self._grammar_value_mapping, self._lemmatize_rule_mapping,
                     word, grammar_val_index, lemmatize_rule_index, freq)

    @property
    def grammemes_count(self) -> int:
        return self._grammemes_matrix.shape[1]

    @property
    def lemmatize_rule_mapping(self) -> Sequence[Tuple[int, str]]:
        return self._lemmatize_rule_mapping


def main():
    import time

    start_time = time.time()
    morph = MorphoAnalyser('RussianDict.zip')
    print('Dictionary loaded in {:.2f} ms'.format((time.time() - start_time) * 1000))

    parses = morph.analyse_word('берегу')
    for parse in parses:
        print(parse, 'Lemma - ', parse.lemma)

    print(morph.build_grammemes_vector('берегу'))


if __name__ == '__main__':
    main()
