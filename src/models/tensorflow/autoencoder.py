import tensorflow as tf

from utils import *

class ConvAutoencoder(object):
    def __init__(self, n_filters=[1, 10, 10, 10], filter_sizes=[3, 3, 3, 3]):
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes

        self.shapes = []
        self.encoder = {'weights': [], 'biases': []}
        self.decoder = {'weights': [], 'biases': []}

        self.x = None
        self.z = None
        self.y = None
        self.loss = None

    def prepare(self, image, target=None):
        current_input = image
        self.x = image

        # Build the encoder
        for layer_i, n_output in enumerate(self.n_filters[1:]):
            n_output *= image.shape[3].value
            n_input = current_input.shape[3].value
            self.shapes.append(current_input.shape.as_list())
            W = tf.Variable(
                tf.random_uniform([
                    self.filter_sizes[layer_i],
                    self.filter_sizes[layer_i],
                    n_input, n_output],
                    -1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([n_output]))
            self.encoder['weights'].append(W)
            self.encoder['biases'].append(b)
            output = lrelu(
                tf.add(tf.nn.conv2d(
                    current_input, W,
                    strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        # Inner-most latent representation
        self.z = current_input

        # Build the decoder using the same weights
        weights = list(reversed(self.encoder['weights']))
        shapes = list(reversed(self.shapes))
        for layer_i, shape in enumerate(shapes):
            W = weights[layer_i]
            b = tf.Variable(tf.zeros([W.shape[2].value]))
            self.decoder['weights'].append(W)
            self.decoder['biases'].append(b)
            output = lrelu(tf.add(
                tf.nn.conv2d_transpose(
                    current_input, W, shape,
                    strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        self.y = output

        if target is not None:
            # Euclidean loss
            self.loss = tf.reduce_sum(tf.square(self.y - target))
        else:
            self.loss = None

