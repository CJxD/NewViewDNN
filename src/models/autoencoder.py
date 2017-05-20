import tensorflow as tf

from utils import *

class ConvAutoencoder(object):
    def __init__(self, n_filters=[1, 10, 10, 10], filter_sizes=[3, 3, 3, 3]):
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes

        self.shapes = []
        self.encoder = {'weights': [], 'biases': []}
        self.decoder = {'weights': [], 'biases': []}

        self._x = None
        self._y = None
        self._z = None
        self._t = None
        self.loss = None

        # Initialise weights
        for i, n_output in enumerate(self.n_filters[1:]):
            n_input = self.n_filters[i]
            
            # Initialise weights
            encW = tf.Variable(
                tf.random_uniform([
                    self.filter_sizes[i],
                    self.filter_sizes[i],
                    n_input, n_output],
                    -1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input)))

            encb = tf.Variable(tf.zeros([n_output]))
 
            decW = encW

            decb = tf.Variable(tf.zeros([n_input]))

            self.encoder['weights'].append(encW)
            self.encoder['biases'].append(encb)

            self.decoder['weights'].insert(0, decW)
            self.decoder['biases'].insert(0, decb)

    def prepare(self, images, targets=None):
        self._x = images
        self._t = targets
        self.shapes = []
        current_input = self._x

        # Build the encoder
        for i, _ in enumerate(self.n_filters[1:]):
            self.shapes.append(current_input.shape.as_list())
            W = self.encoder['weights'][i]
            b = self.encoder['biases'][i]

            conv = tf.nn.conv2d(
                     current_input, W,
                     strides=[1, 2, 2, 1], padding='SAME')
            output = lrelu(tf.add(conv, b))
            current_input = output

        # Inner-most latent representation
        self._z = current_input

        # Build the decoder
        for i, shape in enumerate(self.shapes[::-1]):
            W = self.decoder['weights'][i]
            b = self.decoder['biases'][i]

            deconv = tf.nn.conv2d_transpose(
                       current_input, W, shape,
                       strides=[1, 2, 2, 1], padding='SAME')
            output = lrelu(tf.add(deconv, b))
            current_input = output

        self._y = output

        if self._t is not None:
            # Euclidean loss
            self.loss = tf.reduce_sum(tf.square(self._y - self._t))
        else:
            self.loss = None

        return self

    @property
    def input(self):
        return self._x

    @input.setter
    def input(self, x):
        self.prepare(x, self._t)

    @property
    def targets(self):
        return self._t

    @targets.setter
    def targets(self, t):
        self.prepare(self._x, t)

    @property
    def output(self):
        return self._y

    @property
    def latent_state(self):
        return self._z

