# -*- coding: utf-8 -*-

import os
from os.path import commonprefix
from collections import defaultdict
from zipfile import ZipFile
from tempfile import TemporaryDirectory
import pickle
from typing import Mapping, Iterable, Tuple, List

import dawg
import numpy as np

class DictionaryBuilder:
    def __init__(self, source_path: str, lang: str, encoding: str='utf-16'):
        self._lang = lang
        self._categories = []
        self._grammar_values = defaultdict(lambda: len(self._grammar_values))
        self._lemmatize_rules = defaultdict(lambda: len(self._lemmatize_rules))
        self._quantized_freqs_mapping = []

        self._grammemes_mappings = self._collect_grammemes_mapping(source_path, encoding)
        self._words = self._collect_words(source_path, encoding)
        self._dawg = self._build_dawg()

    def _collect_grammemes_mapping(self, source_path: str, encoding: str) -> List[Mapping]:
        with open(source_path, encoding=encoding) as f:
            self._categories = f.readline().strip().split('\t')[1].split(';')
            grammemes = [set() for _ in range(len(self._categories))]
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                cur_grammemes = line.strip().split('\t')[1].split(';')
                for i, grammeme in enumerate(cur_grammemes):
                    grammemes[i].add(grammeme)

        grammemes_mappings = []
        for grammemes_list in grammemes:
            grammemes_mappings.append({grammeme: i for i, grammeme in enumerate(sorted(grammemes_list))})

        return grammemes_mappings

    def _convert_grammar_val_to_vector(self, grammar_value: str) -> np.ndarray:
        grammar_value_vector = self._get_empty_grammar_val_vector()
        for i, (grammeme, mapping) in enumerate(zip(grammar_value.split(';'), self._grammemes_mappings)):
            grammar_value_vector[i] = mapping[grammeme]
        return grammar_value_vector

    def _get_empty_grammar_val_vector(self) -> np.ndarray:
        return np.zeros(len(self._grammemes_mappings), dtype=np.int)

    def _collect_words(self, source_path: str, encoding: str) -> Iterable[Tuple[str, Tuple[int, int, float]]]:
        words = []
        with open(source_path, encoding=encoding) as f:
            f.readline()

            is_start_record = False
            lexeme = ''
            base_grammar_val_vector = self._get_empty_grammar_val_vector()
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    is_start_record = True
                    continue
                line = line.strip()
                if is_start_record:
                    lexeme, grammar_value, _ = line.split('\t')
                    base_grammar_val_vector = self._convert_grammar_val_to_vector(grammar_value)
                    is_start_record = False
                    continue

                word, grammar_val, freq = line.split('\t')

                grammar_val = tuple(self._convert_grammar_val_to_vector(grammar_val) + base_grammar_val_vector)
                freq = float(freq)

                common_prefix = commonprefix([word, lexeme])
                lemmatize_rule = (len(word) - len(common_prefix), lexeme[len(common_prefix):])

                words.append((word, (self._grammar_values[grammar_val], self._lemmatize_rules[lemmatize_rule], freq)))
        return words

    def _build_dawg(self) -> dawg.RecordDAWG:
        words = self._filter_words(self._words)
        freqs = set(freq for _, (_, _, freq) in words)
        freq_to_index = self._quantize_freqs(freqs)
        words = ((word, (freq_to_index[freq], lem_rule, gr_val)) for word, (gr_val, lem_rule, freq) in words)

        return dawg.RecordDAWG('>HHH', words)

    def _filter_words(self, words: Iterable[Tuple[str, Tuple[int, int, float]]]):
        base_dawg = dawg.RecordDAWG('>HHf', words)

        filtered_words = []
        for key in set(base_dawg.keys()):
            values = sorted(base_dawg.get_value(key), key=lambda x: x[0])
            prev_val, prev_lemmatize_rule = None, None
            prev_val_freq = 0.
            for val, lemmatize_rule, freq in values:
                if val == prev_val and lemmatize_rule == prev_lemmatize_rule:
                    prev_val_freq += freq
                else:
                    if prev_val is not None:
                        filtered_words.append((key, (prev_val, prev_lemmatize_rule, prev_val_freq)))
                    prev_val, prev_lemmatize_rule, prev_val_freq = val, lemmatize_rule, freq
            if prev_val is not None:
                filtered_words.append((key, (prev_val, prev_lemmatize_rule, prev_val_freq)))

        return filtered_words

    def _quantize_freqs(self, freqs: Iterable[float], power: int=12):
        freqs = np.array(sorted(freqs))
        hist = np.histogram(np.log10(freqs[1:]), bins=2 ** power - 1)

        quantized_freqs_indices = np.zeros_like(freqs, dtype=np.int)
        self._quantized_freqs_mapping.append(0.)
        cur_pos = 1
        for i, (count, val) in enumerate(zip(*hist)):
            quantized_freqs_indices[cur_pos: cur_pos + count] = i + 1
            self._quantized_freqs_mapping.append(10 ** val)
            cur_pos += count

        freq_to_index = {freq: quantized_freqs_indices[i] for i, freq in enumerate(freqs)}
        return freq_to_index

    def save(self, path: str):
        def mapping_to_tuple(mapping: Mapping[str, int]):
            return tuple(val for val, _ in sorted(mapping.items(), key=lambda x: x[1]))

        grammemes_mappings = [mapping_to_tuple(mapping) for mapping in self._grammemes_mappings]
        grammar_value_mapping = mapping_to_tuple(self._grammar_values)
        lemmatize_rule_mapping = mapping_to_tuple(self._lemmatize_rules)
        alphabet = tuple(sorted(set(symb for word, _ in self._words for symb in word)))
        similar_letters = (('е', 'ё'), ) if self._lang == 'Russian' else ()
        quantized_freqs_mapping = tuple(self._quantized_freqs_mapping)

        with TemporaryDirectory(suffix='dict') as temp_dir:
            self._dawg.save(os.path.join(temp_dir, 'dict.dawg'))

            with open(os.path.join(temp_dir, 'dict.info'), 'wb') as f:
                pickle.dump((self._categories, grammemes_mappings, grammar_value_mapping, lemmatize_rule_mapping,
                             alphabet, similar_letters, quantized_freqs_mapping),
                            file=f, protocol=pickle.HIGHEST_PROTOCOL)

            with ZipFile(path, 'w') as result_file:
                result_file.write(os.path.join(temp_dir, 'dict.dawg'), arcname='dict.dawg')
                result_file.write(os.path.join(temp_dir, 'dict.info'), arcname='dict.info')


def main():
    builder = DictionaryBuilder('RussianDict.txt', lang='Russian')
    builder.save('RussianDict.zip')


if __name__ == '__main__':
    main()
