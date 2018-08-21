# -*- coding: utf-8 -*-

import os

import numpy as np

from neuromorphy.dictionary import MorphoAnalyser
from neuromorphy.train.data_info import DataInfoBuilder, DataInfo
from neuromorphy.train.tagger_model import TaggerModel
from neuromorphy.train.batch_generator import BatchGenerator
from neuromorphy.train.corpus_iterator import CorpusIterator


class ModelTrainer:
    def __init__(self, model_path: str, dict_path: str, train_path: str,
                 val_path: str, test_path: str=None, is_cuda=False, is_train=True):
        self._model_path = model_path

        self._morph = MorphoAnalyser(dict_path)

        if os.path.isdir(model_path):
            self._data_info = DataInfo.restore(self._morph, model_path)
        else:
            self._data_info = DataInfoBuilder(self._morph, train_path, val_path, test_path).build()

        if is_train:
            train_generator = BatchGenerator(data_info=self._data_info,
                                             corpus_iterator=CorpusIterator(train_path))
            val_generator = BatchGenerator(data_info=self._data_info,
                                           corpus_iterator=CorpusIterator(val_path))
        else:
            train_generator, val_generator = None, None

        # TODO: construct model from config
        # TODO: consider passing data_info directly to the model
        self._model = TaggerModel(is_train=is_train,
                                  data_info=self._data_info,
                                  word_to_index=None,
                                  word_embeddings=None,
                                  is_cuda=is_cuda,
                                  word_emb_dim=200,
                                  train_generator=train_generator,
                                  val_generator=val_generator)

        if os.path.isdir(model_path):
            self._model.restore(model_path)

    def fit(self, epochs_count=200):
        try:
            self._model.fit(epochs_count=epochs_count)
        except KeyboardInterrupt:
            pass

        self._model.save(self._model_path)
        self._data_info.save(self._model_path)

    def predict(self, sentence):
        chars = np.zeros((1, len(sentence), self._data_info._max_word_len))
        grammemes = np.zeros((1, len(sentence), self._data_info.grammemes_matrix.shape[-1]))

        for i, word in enumerate(sentence):
            chars[0, i, -len(word):] = \
                [self._data_info.get_char_index(char) for char in word[-self._data_info._max_word_len:]]
            grammemes[0, i] = self._morph.build_grammemes_vector(word)

        grammar_val_indices, lemma_indices = self._model.predict(chars, grammemes)

        print([self._morph.lemmatize_rule_mapping[ind - 1] for ind in lemma_indices[0]])

        for grammar_val_ind, lemma_ind, word in zip(grammar_val_indices[0], lemma_indices[0], sentence):
            cut, append = self._morph.lemmatize_rule_mapping[lemma_ind - 1]
            print(cut, append)
            lemma = (word[:-cut] if cut != 0 else word) + append
            print(word, lemma, self._data_info.labels[grammar_val_ind])


def main():
    trainer = ModelTrainer(model_path='Model',
                           dict_path='RussianDict.zip',
                           train_path='UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu',
                           val_path='UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu',
                           test_path='UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu',
                           is_train=True)
    # trainer.fit(100)

    trainer.predict(["Приемная", "была", "обставлена", "скромно"])


if __name__ == '__main__':
    main()