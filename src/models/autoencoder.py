import tensorflow as tf

from utils import *

activation_fn = lrelu

class ConvAutoencoder(object):
    def __init__(self, filter_sizes=[3, 3, 3], n_filters=[10, 10, 10], image_channels=1):
        self.n_filters = [image_channels] + n_filters
        self.filter_sizes = filter_sizes

        self.weights = {}
        self.biases = {}

        self._x = None
        self._y = None
        self._z = None
        self._t = None
        self.loss = None

        # Initialise weights
        for i, n_output in enumerate(self.n_filters[1:]):
            n_input = self.n_filters[i]
            
            # Initialise kernel weights
            id = str(i+1)
            self.weights['conv_' + id] = tf.Variable(
                tf.random_uniform([
                    self.filter_sizes[i],
                    self.filter_sizes[i],
                    n_input, n_output],
                    -1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input)))

            # Initialise convolution biases
            self.biases['conv_' + id] = tf.Variable(tf.zeros([n_output]))
            self.biases['deconv_' + id] = tf.Variable(tf.zeros([n_input]))

    def build(self, images, targets=None):
        self._x = images
        self._t = targets
        shapes = []
        prev_layer = self._x

        # Build the encoder
        for i, _ in enumerate(self.n_filters[1:]):
            id = str(i+1)
            shapes.append(prev_layer.shape.as_list())
            prev_layer = self.conv_layer(prev_layer, id)

        # Inner-most latent representation
        self._z = prev_layer

        # Build the decoder
        for i, shape in enumerate(shapes[::-1]):
            id = str(len(shapes) - i)
            prev_layer = self.deconv_layer(prev_layer, shape, id)

        self._y = prev_layer

        if self._t is not None:
            # Euclidean loss
            self.loss = tf.reduce_sum(tf.square(self._y - self._t))
        else:
            self.loss = None

        return self

    def avg_pool(self, bottom, id):
        name = 'pool_' + id
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, id):
        name = 'pool_' + id
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, id):
        name = 'conv_' + id
        with tf.variable_scope(name):
            kernel = self.get_kernel(name)
            bias = self.get_bias(name)

            conv = tf.nn.conv2d(bottom, kernel, strides=[1, 2, 2, 1], padding='SAME')
            biased = tf.nn.bias_add(conv, bias)
            activation = activation_fn(biased)

            return activation

    def deconv_layer(self, bottom, top_shape, id):
        name = 'deconv_' + id
        with tf.variable_scope(name):
            # Use same kernel as encoder
            kernel = self.get_kernel('conv_' + id)
            bias = self.get_bias(name)

            deconv = tf.nn.conv2d_transpose(bottom, kernel, top_shape, strides=[1, 2, 2, 1], padding='SAME')
            biased = tf.nn.bias_add(deconv, bias)
            activation = activation_fn(biased)

            return activation

    def get_kernel(self, name):
        return self.weights[name]

    def get_bias(self, name):
        return self.biases[name]

    @property
    def input(self):
        return self._x

    @input.setter
    def input(self, x):
        self.build(x, self._t)

    @property
    def targets(self):
        return self._t

    @targets.setter
    def targets(self, t):
        self.build(self._x, t)

    @property
    def output(self):
        return self._y

    @property
    def latent_state(self):
        return self._z

