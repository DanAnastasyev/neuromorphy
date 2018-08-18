# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnCompatibleLSTMCell, CudnnLSTM
from tensorflow.contrib.rnn import DropoutWrapper

from neuromorphy.train.word_embedding_model import WordEmbeddingsModel


class TaggerModel:
    def __init__(self, chars_count, chars_matrix, grammemes_matrix, word_emb_dim, labels_count, lemma_labels_count,
                 train_generator, val_generator=None, is_cuda=False, rnn_dim=128,
                 word_to_index=None, word_embeddings=None):

        with tf.Graph().as_default() as graph:
            self._word_embedding_model = \
                self._build_word_embedding_model(chars_count, chars_matrix, word_emb_dim,
                                                 word_to_index, word_embeddings, graph)

            words, labels, lemma_labels = self._build_dataset_iterators(train_generator, val_generator)

            self._is_training = tf.placeholder_with_default(False, shape=())
            self._loss = 0.
            self._accuracies = []
            self._reset_ops = []

            word_embedding = self._word_embedding_model(words)
            with tf.variable_scope('grammemes_input'):
                grammemes_matrix = tf.Variable(initial_value=grammemes_matrix, trainable=False,
                                               name='grammemes_matrix', dtype=tf.float32)

                grammemes_input = tf.nn.embedding_lookup(grammemes_matrix, words, name='grammemes_vector_lookup')
                grammemes_input = tf.layers.dropout(grammemes_input, 0.2, training=self._is_training,
                                                    noise_shape=tf.concat([tf.shape(grammemes_input)[:-1], [1]],
                                                                          axis=0))
                grammemes_embedding = tf.layers.dense(grammemes_input, 64, activation='elu')

            with tf.variable_scope('encoder'):
                outputs = tf.concat([word_embedding, grammemes_embedding], axis=-1)
                outputs = self._build_rnn('bilstm-1', is_cuda, rnn_dim, outputs,
                                          state_dropout_rate=0.2, output_dropout_rate=0.2)

                self._build_pos_lm(labels, labels_count, outputs, rnn_dim)

                outputs = self._build_rnn('bilstm-2', is_cuda, rnn_dim, outputs,
                                          state_dropout_rate=0.2, output_dropout_rate=0.15)

            self._build_output('grammar_vals', outputs, labels, labels_count, proj_dim=128)
            self._build_output('lemmas', outputs, lemma_labels, lemma_labels_count, proj_dim=128)

            with tf.variable_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer()

                grads, variables = zip(*optimizer.compute_gradients(self._loss))
                grads, norms = tf.clip_by_global_norm(grads, 5.)
                self._train_step = optimizer.apply_gradients(zip(grads, variables),
                                                             tf.train.get_or_create_global_step())

            self._sess = tf.Session(graph=graph)
            self._sess.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()

    def _build_word_embedding_model(self, chars_count, chars_matrix, word_emb_dim,
                                    word_to_index, word_embeddings, graph):
        if word_embeddings is not None:
            known_words = set(word_embeddings.wv.index2word) & set(word_to_index.keys())
            word2index_in = {word: word_to_index[word] for word in known_words}
            word2index_out = {word: (i, word_embeddings.wv.vocab[word].index) for i, word in enumerate(known_words)}

            data = np.array([word2index_in[word] for word in known_words])
            labels = np.array([word2index_out[word][0] for word in known_words])

            embeddings = np.zeros((len(word2index_out), word_embeddings.wv.vectors.shape[1]))
            for _, (index, word_vectors_index) in word2index_out.items():
                embeddings[index] = word_embeddings.wv.vectors[word_vectors_index]

            assert word_emb_dim == embeddings.shape[1]
        else:
            data, labels, embeddings = None, None, None

        word_emb_model = WordEmbeddingsModel(chars_count=chars_count,
                                             chars_matrix=chars_matrix,
                                             output_dim=word_emb_dim,
                                             train_data=data, train_labels=labels,
                                             word_embeddings=embeddings,
                                             graph=graph)
        return word_emb_model

    def _build_dataset_iterators(self, train_generator, val_generator):
        self._train_batchs_count = train_generator.batchs_count
        train_dataset = train_generator.make_dataset()

        dataset_iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                       train_dataset.output_shapes)
        self._train_init_op = dataset_iter.make_initializer(train_dataset)

        if val_generator is not None:
            self._val_batchs_count = val_generator.batchs_count
            self._val_init_op = dataset_iter.make_initializer(val_generator.make_dataset())
        else:
            self._val_batchs_count = 0
            self._val_init_op = None

        return dataset_iter.get_next()

    def _build_rnn(self, name, is_cuda, rnn_dim, inputs, state_dropout_rate, output_dropout_rate):
        with tf.variable_scope(name):
            if is_cuda:
                lstm_cell = CudnnLSTM(num_layers=1, num_units=rnn_dim, direction='bidirectional')
                outputs, _ = lstm_cell(inputs)
            else:
                state_keep_prob = 1. - state_dropout_rate * tf.cast(self._is_training, tf.float32)
                single_cell = lambda: DropoutWrapper(CudnnCompatibleLSTMCell(rnn_dim),
                                                     state_keep_prob=state_keep_prob,
                                                     variational_recurrent=True,
                                                     input_size=inputs.get_shape()[-1],
                                                     dtype=tf.float32)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    single_cell(), single_cell(), inputs, time_major=True, dtype=tf.float32)
                outputs = tf.concat(outputs, axis=-1)
        outputs = tf.layers.dropout(outputs, output_dropout_rate, training=self._is_training,
                                    noise_shape=tf.concat([[1], tf.shape(outputs)[1:]], axis=0))
        return outputs

    def _build_pos_lm(self, labels, labels_count, inputs, rnn_dim):
        padding_vector = tf.cast(tf.fill([1, tf.shape(labels)[1]], value=labels_count), dtype=tf.int64)
        shift_forward_labels = tf.concat([labels[1:], padding_vector], axis=0)
        forward_output = inputs[:, :, :rnn_dim]
        self._build_output('forward', forward_output, shift_forward_labels, labels_count + 1, proj_dim=128)

        shift_backward_labels = tf.concat([padding_vector, labels[:-1]], axis=0)
        backward_output = inputs[:, :, rnn_dim:]
        self._build_output('backward', backward_output, shift_backward_labels, labels_count + 1, proj_dim=128)

    def _build_output(self, name, inputs, labels, labels_count, proj_dim=-1):
        with tf.variable_scope(name + '_output'):
            if proj_dim != -1:
                inputs = tf.layers.dense(inputs, units=proj_dim, activation=tf.nn.relu, name='output_proj')
                inputs = tf.layers.dropout(inputs, 0.2, training=self._is_training)
            logits = tf.layers.dense(inputs, labels_count, name='output')
            preds = tf.argmax(logits, axis=-1)

        with tf.variable_scope(name + '_loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            mask = tf.cast(tf.not_equal(labels, 0), dtype=tf.float32)
            non_zeros_count = tf.reduce_sum(mask)
            self._loss += tf.reduce_sum(losses * mask) / non_zeros_count

        with tf.variable_scope(name + '_accuracy'):
            correct_count = tf.get_variable(name='correct_count', shape=(), dtype=tf.float32,
                                            initializer=tf.zeros_initializer(), trainable=False)
            total_count = tf.get_variable(name='total_count', shape=(), dtype=tf.float32,
                                          initializer=tf.zeros_initializer(), trainable=False)

            correct_count = \
                correct_count.assign_add(tf.reduce_sum(tf.cast(tf.equal(preds, labels), dtype=tf.float32) * mask))
            total_count = total_count.assign_add(non_zeros_count)
            accuracy = tf.cond(tf.not_equal(total_count, 0.),
                               lambda: correct_count / total_count,
                               lambda: 0.)

            self._accuracies.append(accuracy)
            self._reset_ops.append(tf.assign(correct_count, 0.))
            self._reset_ops.append(tf.assign(total_count, 0.))

    def fit(self, epochs_count=100):
        if self._word_embedding_model.can_be_pretrained:
            self._word_embedding_model.fit(500)

        for epoch in range(epochs_count):
            self._sess.run(self._train_init_op)
            self._run_epoch(epoch, epochs_count, is_train=True)

            if self._val_init_op is not None:
                self._sess.run(self._val_init_op)
                self._run_epoch(epoch, epochs_count, is_train=False)

    def _run_epoch(self, epoch, epochs_count, is_train):
        total_loss = 0.
        batchs_count = self._train_batchs_count if is_train else self._val_batchs_count
        name = 'Train' if is_train else 'Val'

        start_time = time.time()
        train_step = [self._train_step] if is_train else []
        grammar_val_acc, lemma_acc, fw_acc, bw_acc = 0., 0., 0., 0.
        for i in range(batchs_count):
            loss, fw_acc, bw_acc, grammar_val_acc, lemma_acc = \
                self._sess.run([self._loss] + self._accuracies + train_step,
                               feed_dict={self._is_training: is_train})[:5]
            total_loss += loss

            print('\rBatch = {} / {}. Loss = {:.4f}. '
                  'Acc = {:.2%}. Lemma Acc = {:.2%}. Forward Acc = {:.2%}. Backward Acc = {:.2%}'.format(
                i, batchs_count, loss, grammar_val_acc, lemma_acc, fw_acc, bw_acc), end='')

        print('\r' + (' ' * 180), end='')
        print('\rEpoch = {} / {}. Time = {:05.2f} s. {:>5s} Loss = {:.4f}. '
              'Acc = {:.2%}. Lemma Acc = {:.2%}. Forward Acc = {:.2%}. Backward Acc = {:.2%}'.format(
            epoch + 1, epochs_count, time.time() - start_time, name,
            total_loss / batchs_count, grammar_val_acc, lemma_acc, fw_acc, bw_acc))

        self._sess.run(self._reset_ops)
        return total_loss / batchs_count, grammar_val_acc, lemma_acc

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        self._saver.save(self._sess, os.path.join(path, 'model.chkp'))

    def restore(self, path):
        self._saver.restore(self._sess, os.path.join(path, 'model.chkp'))
