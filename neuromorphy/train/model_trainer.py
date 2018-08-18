# -*- coding: utf-8 -*-

import os

from neuromorphy.dictionary import MorphoAnalyser
from neuromorphy.train.data_info_builder import DataInfoBuilder
from neuromorphy.train.tagger_model import TaggerModel
from neuromorphy.train.batch_generator import BatchGenerator
from neuromorphy.train.corpus_iterator import CorpusIterator


class ModelTrainer:
    def __init__(self, dict_path: str, train_path: str, val_path: str, test_path: str=None, is_cuda=False):
        morph = MorphoAnalyser(dict_path)
        self._data_info = DataInfoBuilder(morph, train_path, val_path, test_path).build()

        train_generator = BatchGenerator(data_info=self._data_info,
                                         corpus_iterator=CorpusIterator(train_path))

        val_generator = BatchGenerator(data_info=self._data_info,
                                       corpus_iterator=CorpusIterator(val_path))

        # TODO: construct model from config
        # TODO: consider passing data_info directly to the model
        self._model = TaggerModel(chars_count=self._data_info.chars_count,
                                  chars_matrix=self._data_info.chars_matrix,
                                  grammemes_matrix=self._data_info.grammemes_matrix,
                                  word_to_index=None,
                                  word_embeddings=None,
                                  is_cuda=is_cuda,
                                  word_emb_dim=200,
                                  labels_count=self._data_info.labels_count,
                                  lemma_labels_count=self._data_info.lemma_labels_count,
                                  train_generator=train_generator,
                                  val_generator=val_generator)

    def fit(self, model_path, epochs_count=200):
        if os.path.isdir(model_path):
            self._model.restore(model_path)
        try:
            self._model.fit(epochs_count=epochs_count)
        except KeyboardInterrupt:
            pass

        self._model.save(model_path)
        self._data_info.save(model_path)


def main():
    trainer = ModelTrainer(dict_path='RussianDict.zip',
                           train_path='UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu',
                           val_path='UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu',
                           test_path='UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu')

    trainer.fit(model_path='Model', epochs_count=1)


if __name__ == '__main__':
    main()