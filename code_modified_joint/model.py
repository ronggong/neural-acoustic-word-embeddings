from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from triplet_loss import triplet_loss


class LSTM(object):
    """LSTM class to set up model."""

    def __init__(self, is_train, n, h, p, scope):
        """Initialize lstm hparams."""

        self.n = n  # -- number of layers
        self.h = h  # -- number of hidden units
        self.p = p if is_train else 0.0  # -- keep
        self.scope = scope  # -- variable scope

    def run(self, x, ts, bidirectional=False, reuse=None, output_shape=None):
        """Run model."""

        with tf.variable_scope(self.scope, reuse=reuse):
            for l in range(self.n):
                with tf.variable_scope("l{}".format(l + 1), reuse=reuse):
                    fw = tf.nn.rnn_cell.LSTMCell(self.h)
                    if l < self.n - 1 and self.p > 0.0:
                        fw = tf.nn.rnn_cell.DropoutWrapper(fw, output_keep_prob=self.p)

                    if bidirectional:
                        bw = tf.nn.rnn_cell.LSTMCell(self.h)
                        if l < self.n - 1 and self.p > 0.0:
                            bw = tf.nn.rnn_cell.DropoutWrapper(bw, output_keep_prob=self.p)

                        x, state = tf.nn.bidirectional_dynamic_rnn(fw, bw, x, sequence_length=ts, dtype=tf.float32)
                        x = tf.concat(x, 2)
                    else:
                        x, state = tf.nn.dynamic_rnn(fw, x, sequence_length=ts, dtype=tf.float32)

            rnn_state = tf.concat([direction.h for direction in state], 1) if bidirectional else state.h

            if isinstance(output_shape, list):
                dense_output_0 = tf.layers.dense(inputs=rnn_state, units=output_shape[0], activation=None)
                dense_output_1 = tf.layers.dense(inputs=rnn_state, units=output_shape[1], activation=None)
                dense_output = [dense_output_0, dense_output_1]
            else:
                # dense layer with linear activation
                dense_output = tf.layers.dense(inputs=rnn_state, units=output_shape, activation=None)

            return dense_output


class Model(object):
    """Neural embedding model."""

    def __init__(self, is_train, config, reuse):
        """Initialize model."""

        self.x = tf.placeholder(tf.float32, [None, None, config.feature_dim])
        self.ts = tf.placeholder(tf.int32, [None])
        self.same_partition = tf.placeholder(tf.int32, [None])
        self.diff_partition = tf.placeholder(tf.int32, [None])

        self.lstm = LSTM(is_train=is_train, n=config.num_layers,
                         h=config.hidden_size, p=config.keep_prob, scope="lstm")

        self.embeddings = self.lstm.run(self.x,
                                        self.ts,
                                        bidirectional=config.bidirectional,
                                        reuse=reuse,
                                        output_shape=config.output_shape)

        if is_train == "train" or is_train == "dev":
            if isinstance(config.output_shape, list):
                if config.mtl == "phn" or config.mtl == "both":
                    embeddings_diff_phn_0, embeddings_diff_phn_1, embeddings_diff_phn_2, embeddings_diff_phn_3 = \
                        self.logit_split_phn(self.embeddings[0], config.batch_size, config.max_diff)
                    loss_phn_0 = triplet_loss(embeddings_diff_phn_0,
                                              self.same_partition, self.diff_partition, margin=config.margin)
                    loss_phn_1 = triplet_loss(embeddings_diff_phn_1,
                                              self.same_partition, self.diff_partition, margin=config.margin)
                    loss_phn_2 = triplet_loss(embeddings_diff_phn_2,
                                              self.same_partition, self.diff_partition, margin=config.margin)
                    loss_phn_3 = triplet_loss(embeddings_diff_phn_3,
                                              self.same_partition, self.diff_partition, margin=config.margin)

                if config.mtl == "pro" or config.mtl == "both":
                    embeddings_diff_pro_0, embeddings_diff_pro_1, embeddings_diff_pro_2, embeddings_diff_pro_3 = \
                        self.logit_split_pro(self.embeddings[1], config.batch_size, config.max_diff)
                    loss_pro_0 = triplet_loss(embeddings_diff_pro_0,
                                              self.same_partition, self.diff_partition, margin=config.margin)
                    loss_pro_1 = triplet_loss(embeddings_diff_pro_1,
                                              self.same_partition, self.diff_partition, margin=config.margin)
                    loss_pro_2 = triplet_loss(embeddings_diff_pro_2,
                                              self.same_partition, self.diff_partition, margin=config.margin)
                    loss_pro_3 = triplet_loss(embeddings_diff_pro_3,
                                              self.same_partition, self.diff_partition, margin=config.margin)

                if config.mtl == "phn":
                    self.loss = loss_phn_0 + loss_phn_1 + loss_phn_2 + loss_phn_3
                elif config.mtl == "pro":
                    self.loss = loss_pro_0 + loss_pro_1 + loss_pro_2 + loss_pro_3
                elif config.mtl == "both":
                    self.loss = loss_phn_0 + loss_phn_1 + loss_phn_2 + loss_phn_3 + \
                                loss_pro_0 + loss_pro_1 + loss_pro_2 + loss_pro_3
                else:
                    raise ValueError("config.mtl {} doesn't exist.".format(config.mtl))
            else:
                self.loss = triplet_loss(self.embeddings, self.same_partition, self.diff_partition,
                                         margin=config.margin)
        else:
            pass

        if is_train == "train":
            optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
            self.optim = optimizer.minimize(self.loss)
        else:
            pass

    @staticmethod
    def logit_split_phn(embeddings, batch_size, max_diff):
        """split and merge embeddings phn diff"""
        anchor, same, diff_pro, diff_phn, diff_phn_pro = tf.split(embeddings, [batch_size,
                                                                               batch_size,
                                                                               batch_size * max_diff,
                                                                               batch_size * max_diff, -1])
        embeddings_diff_phn_0 = tf.concat([anchor, same, diff_phn], 0)
        embeddings_diff_phn_1 = tf.concat([anchor, same, diff_phn_pro], 0)
        embeddings_diff_phn_2 = tf.concat([anchor, diff_pro[0:batch_size, :], diff_phn], 0)
        embeddings_diff_phn_3 = tf.concat([anchor, diff_pro[0:batch_size, :], diff_phn_pro], 0)
        return embeddings_diff_phn_0, embeddings_diff_phn_1, embeddings_diff_phn_2, embeddings_diff_phn_3

    @staticmethod
    def logit_split_pro(embeddings, batch_size, max_diff):
        """split and merge embeddings pro diff"""
        anchor, same, diff_pro, diff_phn, diff_phn_pro = tf.split(embeddings, [batch_size,
                                                                               batch_size,
                                                                               batch_size * max_diff,
                                                                               batch_size * max_diff, -1])
        embeddings_diff_pro_0 = tf.concat([anchor, same, diff_pro], 0)
        embeddings_diff_pro_1 = tf.concat([anchor, same, diff_phn_pro], 0)
        embeddings_diff_pro_2 = tf.concat([anchor, diff_phn[0:batch_size, :], diff_pro], 0)
        embeddings_diff_pro_3 = tf.concat([anchor, diff_phn[0:batch_size, :], diff_phn_pro], 0)
        return embeddings_diff_pro_0, embeddings_diff_pro_1, embeddings_diff_pro_2, embeddings_diff_pro_3

    def get_loss_train(self, sess, x, ts, same_partition, diff_partition):
        """Calculate loss (for training)."""

        return sess.run([self.optim, self.loss], feed_dict={
            self.x: x,  # input
            self.ts: ts,  # input lengths
            self.same_partition: same_partition,  # -- track same indices
            self.diff_partition: diff_partition  # -- track diff indices
        })

    def get_loss_dev(self, sess, x, ts, same_partition, diff_partition):
        """Calculate loss (for validation)."""

        return sess.run(self.loss, feed_dict={
            self.x: x,  # input
            self.ts: ts,  # input lengths
            self.same_partition: same_partition,  # -- track same indices
            self.diff_partition: diff_partition  # -- track diff indices
        })

    def get_embeddings(self, sess, x, ts):
        """Calculate average precision (for evaluation)."""

        return sess.run(self.embeddings, feed_dict={
            self.x: x,  # input
            self.ts: ts  # input lengths
        })
