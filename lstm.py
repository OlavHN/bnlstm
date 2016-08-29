import math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

class LSTMCell(RNNCell):
    '''Vanilla LSTM implemented with same initializations as BN-LSTM'''
    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # Keep W_xh and W_hh separate here as well to reuse initialization methods
            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [x_size, 4 * self.num_units],
                initializer=orthonogal_initializer())
            W_hh = tf.get_variable('W_hh',
                [self.num_units, 4 * self.num_units],
                initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self.num_units])

            # hidden = tf.matmul(x, W_xh) + tf.matmul(h, W_hh) + bias
            # improve speed by concat.
            concat = tf.concat(1, [x, h])
            W_both = tf.concat(0, [W_xh, W_hh])
            hidden = tf.matmul(concat, W_both) + bias

            i, j, f, o = tf.split(1, 4, hidden)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)

class BNLSTMCell(RNNCell):
    '''Batch normalized LSTM as described in arxiv.org/abs/1603.09025'''
    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [x_size, 4 * self.num_units],
                initializer=orthonogal_initializer())
            W_hh = tf.get_variable('W_hh',
                [self.num_units, 4 * self.num_units],
                initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self.num_units])

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)

            mean_xh, var_xh = tf.nn.moments(xh, [0])
            xh_scale = tf.get_variable('xh_scale', [4 * self.num_units], initializer=tf.constant_initializer(0.1))

            mean_hh, var_hh = tf.nn.moments(hh, [0])
            hh_scale = tf.get_variable('hh_scale', [4 * self.num_units], initializer=tf.constant_initializer(0.1))

            static_offset = tf.constant(0, dtype=tf.float32, shape=[4 * self.num_units])
            epsilon = 1e-3

            bn_xh = tf.nn.batch_normalization(xh, mean_xh, var_xh, static_offset, xh_scale, epsilon)
            bn_hh = tf.nn.batch_normalization(hh, mean_hh, var_hh, static_offset, hh_scale, epsilon)

            hidden = bn_xh + bn_hh + bias

            i, j, f, o = tf.split(1, 4, hidden)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)

            mean_c, var_c = tf.nn.moments(new_c, [0])
            c_scale = tf.get_variable('c_scale', [self.num_units], initializer=tf.constant_initializer(0.1))
            c_offset = tf.get_variable('c_offset', [self.num_units])

            bn_new_c = tf.nn.batch_normalization(new_c, mean_c, var_c, c_offset, c_scale, epsilon)

            new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)

def orthonogal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

def bn_lstm_identity_initializer(scale):
    def _initializer(shape, dtype=tf.float32):
        '''Ugly cause LSTM params calculated in one matrix multiply'''
        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale
        t[:, :size] = orthonogal([size, size])
        t[:, size * 2:size * 3] = orthonogal([size, size])
        t[:, size * 3:] = orthonogal([size, size])
        return tf.constant(t, dtype)

    return _initializer

def orthonogal_initializer():
    def _initializer(shape, dtype=tf.float32):
        return tf.constant(orthonogal(shape), dtype)
    return _initializer
