# -*- coding: utf-8 -*-

from itertools import chain
from typing import Tuple, List
import attr


@attr.s(frozen=True)
class Token:
    token: str = attr.ib()
    lemma: str = attr.ib()
    grammar_value: str = attr.ib()


class CorpusIterator:
    def __init__(self, path: str, separator: str='\t', token_col_index: int=1, lemma_col_index: int=2,
                 grammar_val_col_indices: Tuple=(3, 5), grammemes_separator: str='|',
                 skip_line_prefix: str='#', encoding: str='utf8'):
        """
        Creates iterator over the corpus in conll-like format:
        - each line contains token and its annotations (lemma and grammar value info) separated by ``separator``
        - sentences are separated by empty line

        :param path: path to corpus
        :param separator: separator between fields
        :param token_col_index: index of token field
        :param lemma_col_index: index of lemma field
        :param grammar_val_col_indices: indices of grammar value fields (e.g. POS and morphological tags)
        :param grammemes_separator: separator between grammemes (as in 'Case=Nom|Definite=Def|Gender=Com|Number=Sing')
        :param skip_line_prefix: prefix for comment lines
        :param encoding: encoding of the corpus file
        """
        self._path = path
        self._separator = separator
        self._token_col_index = token_col_index
        self._lemma_col_index = lemma_col_index
        self._grammar_val_col_indices = grammar_val_col_indices
        self._grammemes_separator = grammemes_separator
        self._skip_line_prefix = skip_line_prefix
        self._encoding = encoding

    def __enter__(self):
        self._file = open(self._path, encoding=self._encoding)
        return self

    def __exit__(self, type, value, traceback):
        self._file.close()

    def __iter__(self):
        return self

    def __next__(self) -> List[Token]:
        sentence = []
        for line in self._file:
            line = line.rstrip()
            if line.startswith(self._skip_line_prefix):
                continue
            if len(line) == 0:
                break

            fields = line.split(self._separator)
            token, lemma = fields[self._token_col_index], fields[self._lemma_col_index]
            grammar_value = '|'.join(chain(*(sorted(fields[col_index].split(self._grammemes_separator))
                                             for col_index in self._grammar_val_col_indices)))

            sentence.append(Token(token, lemma, grammar_value))
        if sentence:
            return sentence
        else:
            raise StopIteration
