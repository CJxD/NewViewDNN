from __future__ import division, print_function
from abc import ABC, abstractmethod

import tensorflow as tf
import math

class CNN(ABC):
    def __init__(self):
        self.data = {}

        self.conv_strides = [1, 1, 1, 1]
        self.conv_padding = 'SAME'

        self.deconv_strides = [1, 1, 1, 1]
        self.deconv_padding = 'SAME'

        self.pool_kernel = [1, 2, 2, 1]
        self.pool_strides = [1, 2, 2, 1]
        self.pool_padding = 'SAME'

        self.activation = tf.nn.relu

        self._x = None
        self._y = None
        self._z = None
        self._t = None
        self.loss = None

    @abstractmethod
    def build(self, images, targets=None):
        pass

    def make_kernel(self, size, n_input, n_output):
        return tf.random_uniform(
                size + [n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input))

    def make_bias(self, n):
        return tf.truncated_normal([n], stddev=0.001)

    def get_kernel(self, name):
        return self.data[name][0]

    def get_bias(self, name):
        return self.data[name][1]

    def avg_pool(self, bottom, name='pool'):
        with tf.variable_scope(name):
            return tf.nn.avg_pool(bottom, ksize=self.pool_kernel, strides=self.pool_strides, padding=self.pool_padding)

    def max_pool(self, bottom, name='pool'):
        with tf.variable_scope(name):
            return tf.nn.max_pool(bottom, ksize=self.pool_kernel, strides=self.pool_strides, padding=self.pool_padding)

    def conv_layer(self, bottom, name='conv'):
        with tf.variable_scope(name):
            kernel = tf.Variable(self.get_kernel(name), name='W')
            bias = tf.Variable(self.get_bias(name), name='b')

            conv = tf.nn.conv2d(bottom, kernel, strides=self.conv_strides, padding=self.conv_padding)
            output = self.activation(conv + bias)

            tf.summary.histogram("weights", kernel)
            tf.summary.histogram("biases", bias)
            tf.summary.histogram("activations", output)

            return output

    def deconv_layer(self, bottom, top_shape, name='deconv'):
        with tf.variable_scope(name):
            kernel = tf.Variable(self.get_kernel(name), name='W')
            bias = tf.Variable(self.get_bias(name), name='b')

            deconv = tf.nn.conv2d_transpose(bottom, kernel, top_shape, strides=self.deconv_strides, padding=self.deconv_padding)
            output = self.activation(deconv + bias)

            tf.summary.histogram("weights", kernel)
            tf.summary.histogram("biases", bias)
            tf.summary.histogram("activations", output)

            return output
    
    def weighted_diff(self, images, targets, threshold=0.1, base_weight=0.5):
        '''Generates a binary  image mask with white areas being the differences,
        and dark areas being the similarities.'''
        # Take absolute difference of images
        diff = tf.abs(targets - images)
        # Squash differences into a single greyscale channel
        greyscale = tf.reduce_sum(diff, 3, keep_dims=True)

        # Threshold the greyscale channel into black and white
        # but add a base level to make sure no part of the image
        # is completely ignored
        min_value = tf.zeros_like(greyscale) + base_weight
        max_value = tf.ones_like(greyscale)
        thresholded = tf.where(greyscale > threshold, max_value, min_value)

        return thresholded

    def euclidean_loss(self, tensor, name='loss'):
        with tf.variable_scope(name):
            return tf.reduce_sum(tf.square(tensor))

    def euclidean_mean(self, tensor, name='loss'):
        with tf.variable_scope(name):
            return tf.reduce_mean(tf.square(tensor))

    @property
    def input(self):
        return self._x

    @input.setter
    def input(self, x):
        self.build(x, self._t)

    @property
    def target(self):
        return self._t

    @target.setter
    def target(self, t):
        self.build(self._x, t)

    @property
    def output(self):
        return self._y

    @property
    def latent_state(self):
        return self._z

