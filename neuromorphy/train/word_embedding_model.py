# -*- coding: utf-8 -*-

import time

import numpy as np
import tensorflow as tf

class WordEmbeddingsModel:
    def __init__(self, is_train: bool, chars_count: int, chars_matrix: np.ndarray, grammemes_matrix: np.ndarray,
                 output_dim: int, train_data: np.ndarray=None, train_labels: np.ndarray=None,
                 word_embeddings: np.ndarray=None, graph: tf.Graph=None):

        self._is_train_mode = is_train
        self._chars_count = chars_count
        self._chars_matrix = chars_matrix
        self._grammemes_matrix = grammemes_matrix
        self._output_dim = output_dim

        self._model = None
        self._can_be_pretrained = all(param is not None for param in (train_data, train_labels, word_embeddings))
        self._sess = None
        self._is_training = None
        self._train_batchs_count = None
        self._train_init_op = None
        self._train_step = None
        self._loss = None

        if self._can_be_pretrained:
            self._build_pretrain_model(train_data, train_labels, word_embeddings, graph)

    def _build_pretrain_model(self, train_data, train_labels, word_embeddings, graph):
        with graph.as_default():
            with tf.variable_scope('word_vector_train'):
                train_generator = tf.data.Dataset() \
                                         .from_tensor_slices((train_data, train_labels)) \
                                         .batch(128) \
                                         .shuffle(len(train_data))

                self._train_batchs_count = train_data.shape[0] // 128
                iterator = tf.data.Iterator.from_structure(train_generator.output_types,
                                                           train_generator.output_shapes)

                self._train_init_op = iterator.make_initializer(train_generator)

                data, labels = iterator.get_next()

            self._is_training = tf.placeholder_with_default(False, shape=())

            self._model = self._build_model()
            word_vectors = self._model(data)

            with tf.variable_scope('word_vector_loss'):
                word_vectors = tf.layers.dense(word_vectors, word_embeddings.shape[1])

                word_embeddings_matrix = tf.Variable(initial_value=word_embeddings, trainable=False,
                                                     name='word_embeddings_matrix', dtype=tf.float32)

                target_vectors = tf.gather(word_embeddings_matrix, labels)
                self._loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(target_vectors, word_vectors), axis=-1))

            with tf.variable_scope('word_vector_optimizer'):
                optimizer = tf.train.AdamOptimizer()

                grads, variables = zip(*optimizer.compute_gradients(self._loss))
                grads, norms = tf.clip_by_global_norm(grads, 5.)
                self._train_step = optimizer.apply_gradients(zip(grads, variables),
                                                             tf.train.get_or_create_global_step())

            self._sess = tf.Session(graph=graph)
            self._sess.run(tf.global_variables_initializer())

    def _build_model(self):
        with tf.variable_scope('chars_input'):
            chars_matrix = tf.Variable(initial_value=self._chars_matrix, trainable=False,
                                       name='chars_matrix', dtype=tf.int32)
            chars_embeddings_matrix = tf.get_variable(name='embeddings', shape=[self._chars_count, 24],
                                                      dtype=tf.float32)
            word_embedding_dense_1 = tf.layers.Dense(512, name='dense', activation=tf.nn.relu)
            word_embedding_dense_2 = tf.layers.Dense(128, name='dense_1')

        with tf.variable_scope('grammemes_input'):
            grammemes_matrix = tf.Variable(initial_value=self._grammemes_matrix, trainable=False,
                                           name='grammemes_matrix', dtype=tf.float32)
            grammemes_embedding_dense = tf.layers.Dense(64, name='dense')

        def _apply(data):
            assert self._is_train_mode or isinstance(data, (list, tuple)) and len(data) == 2

            with tf.variable_scope('chars_embedding'):
                if not self._is_train_mode:
                    chars_input = data[0]
                else:
                    chars_input = tf.nn.embedding_lookup(chars_matrix, data, name='chars_vector_lookup')

                chars_embeddings = tf.nn.embedding_lookup(chars_embeddings_matrix, chars_input, name='chars_emb_lookup')

                chars_embeddings = tf.reshape(chars_embeddings, tf.concat([tf.shape(chars_embeddings)[:-2],
                                                                           [24 * chars_matrix.shape[1]]], axis=0))

                chars_embeddings = tf.layers.dropout(chars_embeddings, 0.25, training=self._is_training)
                word_embedding = word_embedding_dense_1(chars_embeddings)

                word_embedding = tf.layers.dropout(word_embedding, 0.25, training=self._is_training)
                word_embedding = word_embedding_dense_2(word_embedding)

            with tf.variable_scope('grammemes_input'):
                if not self._is_train_mode:
                    grammemes_input = data[1]
                else:
                    grammemes_input = tf.nn.embedding_lookup(grammemes_matrix, data, name='grammemes_vector_lookup')
                grammemes_embedding = grammemes_embedding_dense(grammemes_input)

            return tf.concat([word_embedding, grammemes_embedding], axis=-1)

        return _apply

    def __call__(self, data: tf.placeholder):
        if self._model is None:
            self._model = self._build_model()
        return self._model(data)

    def fit(self, epochs_count: int=100):
        assert self._can_be_pretrained, 'You have to provide pretraining data to fit this model'
        with self._sess:
            for epoch in range(epochs_count):
                self._sess.run(self._train_init_op)
                self._run_epoch(epoch, epochs_count, is_train=True)

    def _run_epoch(self, epoch: int, epochs_count: int, is_train: bool):
        total_loss = 0.
        name = 'Train' if is_train else 'Val'

        start_time = time.time()
        train_step = [self._train_step] if is_train else []
        for i in range(self._train_batchs_count):
            loss = self._sess.run([self._loss] + train_step, feed_dict={self._is_training: is_train})[0]

            total_loss += loss

            print('\rBatch = {} / {}. Loss = {:.4f}'.format(
                i, self._train_batchs_count, loss), end='')

        print('\r' + (' ' * 180), end='')
        print('\rEpoch = {} / {}. Time = {:05.2f} s. {:>5s} Loss = {:.4f}'.format(
            epoch + 1, epochs_count, time.time() - start_time, name, total_loss / self._train_batchs_count))

        return total_loss / self._train_batchs_count

    @property
    def can_be_pretrained(self):
        return self._can_be_pretrained
