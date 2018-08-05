# -*- coding: utf-8 -*-

from dictionary.dictionary import MorphoAnalyser
from data_info_builder import DataInfoBuilder
from tagger_model import TaggerModel
from batch_generator import BatchGenerator
from corpus_iterator import CorpusIterator


class ModelTrainer:
    def __init__(self, dict_path: str, train_path: str, val_path: str, test_path: str=None):
        morph = MorphoAnalyser(dict_path)
        data_info = DataInfoBuilder(morph, train_path, val_path, test_path)

        train_generator = BatchGenerator(data_info=data_info,
                                         corpus_iterator=CorpusIterator(train_path))

        val_generator = BatchGenerator(data_info=data_info,
                                       corpus_iterator=CorpusIterator(val_path))

        # TODO: construct model from config
        # TODO: consider passing data_info directly to the model
        self._model = TaggerModel(chars_count=data_info.chars_count,
                                  chars_matrix=data_info.chars_matrix,
                                  grammemes_matrix=data_info.grammemes_matrix,
                                  word_to_index=None,
                                  word_embeddings=None,
                                  word_emb_dim=200,
                                  labels_count=data_info.labels_count,
                                  lemma_labels_count=data_info.lemma_labels_count,
                                  train_generator=train_generator,
                                  val_generator=val_generator)

    def fit(self):
        self._model.fit(epochs_count=200)


def main():
    trainer = ModelTrainer(dict_path='RussianDict.zip',
                           train_path='UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu',
                           val_path='UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu',
                           test_path='UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu')

    trainer.fit()


if __name__ == '__main__':
    main()