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
        self._data_info.save(self._model_path)
        try:
            self._model.fit(epochs_count=epochs_count, save_path=self._model_path)
        except KeyboardInterrupt:
            pass

    def predict(self, sentence):
        chars = np.zeros((1, len(sentence), self._data_info._max_word_len))
        grammemes = np.zeros((1, len(sentence), self._data_info.grammemes_matrix.shape[-1]))

        for i, word in enumerate(sentence):
            chars[0, i, -len(word):] = \
                [self._data_info.get_char_index(char) for char in word[-self._data_info._max_word_len:]]
            grammemes[0, i] = self._morph.build_grammemes_vector(word)

        grammar_val_indices, lemma_indices = self._model.predict(chars, grammemes)

        def get_lemma(word, ind):
            cut, append = self._morph.lemmatize_rule_mapping[ind - 1]
            return (word[:-cut] if cut != 0 else word) + append

        lemmas = [get_lemma(word, ind) for word, ind in zip(sentence, lemma_indices[0])]
        return grammar_val_indices[0], lemmas

        # return [self._data_info.labels[grammar_val_ind] for grammar_val_ind in grammar_val_indices[0]], lemmas

    def predict_batch(self, batch):
        max_sent_len = max(len(sent) for sent in batch)
        chars = np.zeros((len(batch), max_sent_len, self._data_info._max_word_len))
        grammemes = np.zeros((len(batch), max_sent_len, self._data_info.grammemes_matrix.shape[-1]))

        for i, sent in enumerate(batch):
            for j, word in enumerate(sent):
                chars[i, j, -len(word):] = \
                    [self._data_info.get_char_index(char) for char in word[-self._data_info._max_word_len:]]
                grammemes[i, j] = self._morph.build_grammemes_vector(word)

        grammar_val_indices_batch = self._model.predict(chars, grammemes)[0]

        return [[self._data_info.labels[grammar_val_ind] for grammar_val_ind in grammar_val_indices]
                for grammar_val_indices in grammar_val_indices_batch]


def load_pretrained_model():
    return ModelTrainer(model_path='Model', dict_path='RussianDict.zip', train_path=None, val_path=None, is_train=False)


def parse_syntagrus_test():
    import time
    import tqdm

    start_time = time.time()
    model = load_pretrained_model()

    with CorpusIterator('UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu') as corpus_iterator, \
            open('test_preds.txt', 'w', encoding='utf8') as f:
        for sentence in tqdm.tqdm(corpus_iterator):
            tokens = [token.token for token in sentence]
            preds, lemmas = model.predict(tokens)
            for token, grammar_val, lemma in zip(tokens, preds, lemmas):
                pos, grammar_val = grammar_val.split('|', maxsplit=1)
                f.write(token + '\t' + pos + '\t' + grammar_val + '\t' + lemma + '\n')
            f.write('\n')

    print('File processed in {:.1f} s'.format(time.time() - start_time))


def read_corpus(path):
    from nltk.tokenize import sent_tokenize, wordpunct_tokenize

    skip_symbols = {'-', '-', '–', '—', '―', '"', "'", '‘', '’', '“'}
    if os.path.isfile(path):
        paths = [path]
    elif os.path.isdir(path):
        paths = [os.path.join(path, file_name) for file_name in os.listdir(path)]
    else:
        assert False

    for path in paths:
        with open(path, encoding='utf8') as f:
            for line in f:
                line = line.strip()

                if len(line.split()) < 10 or line[0] in skip_symbols:
                    continue

                for sent in sent_tokenize(line):
                    yield wordpunct_tokenize(sent)
                yield []


def parse_plain_text(path):
    import tqdm

    model = load_pretrained_model()
    with open(os.path.splitext(path)[0] + '_parsed.txt', 'w', encoding='utf8') as f:
        for sent in tqdm.tqdm(read_corpus(path)):
            if not sent:
                f.write('\n')
            gram_vals, lemmas = model.predict(sent)
            for grammar_val, lemma in zip(gram_vals, lemmas):
                f.write(lemma + '_' + str(grammar_val) + ' ')


if __name__ == '__main__':
    parse_plain_text('beda.txt')
