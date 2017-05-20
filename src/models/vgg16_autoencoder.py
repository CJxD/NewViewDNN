import tensorflow as tf
import math

from utils import lrelu

activation_fn = tf.nn.relu

class VGG16Autoencoder(object):
    def __init__(self, image_channels=1):
        self.weights = {}
        self.biases = {}

        self._x = None
        self._y = None
        self._z = None
        self._t = None
        self.loss = None
  
        # Initialise convolution kernel weights
        self.weights['conv1_1'] = self.make_kernel([3, 3], image_channels, 64)
        self.weights['conv1_2'] = self.make_kernel([3, 3], 64, 64)

        self.weights['conv2_1'] = self.make_kernel([3, 3], 64, 128)
        self.weights['conv2_2'] = self.make_kernel([3, 3], 128, 128)

        self.weights['conv3_1'] = self.make_kernel([3, 3], 128, 256)
        self.weights['conv3_2'] = self.make_kernel([3, 3], 256, 256)
        self.weights['conv3_3'] = self.make_kernel([3, 3], 256, 256)

        self.weights['conv4_1'] = self.make_kernel([3, 3], 256, 512)
        self.weights['conv4_2'] = self.make_kernel([3, 3], 512, 512)
        self.weights['conv4_3'] = self.make_kernel([3, 3], 512, 512)

        self.weights['conv5_1'] = self.make_kernel([3, 3], 512, 512)
        self.weights['conv5_2'] = self.make_kernel([3, 3], 512, 512)
        self.weights['conv5_3'] = self.make_kernel([3, 3], 512, 512)

        # Initialise convolution biases
        self.biases['conv1_1'] = self.make_bias(64)
        self.biases['conv1_2'] = self.make_bias(64)

        self.biases['conv2_1'] = self.make_bias(128)
        self.biases['conv2_2'] = self.make_bias(128)

        self.biases['conv3_1'] = self.make_bias(256)
        self.biases['conv3_2'] = self.make_bias(256)
        self.biases['conv3_3'] = self.make_bias(256)

        self.biases['conv4_1'] = self.make_bias(512)
        self.biases['conv4_2'] = self.make_bias(512)
        self.biases['conv4_3'] = self.make_bias(512)

        self.biases['conv5_1'] = self.make_bias(512)
        self.biases['conv5_2'] = self.make_bias(512)
        self.biases['conv5_3'] = self.make_bias(512)

        # Initialise deconvolution kernel weights
        self.weights['deconv5_1'] = self.make_kernel([4, 4], 512, 512)
        self.weights['deconv4_1'] = self.make_kernel([4, 4], 256, 512)
        self.weights['deconv3_1'] = self.make_kernel([4, 4], 128, 256)
        self.weights['deconv2_1'] = self.make_kernel([4, 4], 64, 128)
        self.weights['deconv1_1'] = self.make_kernel([4, 4], image_channels, 64)

        # Initialise deconvolution biases
        self.biases['deconv5_1'] = self.make_bias(512)
        self.biases['deconv4_1'] = self.make_bias(256)
        self.biases['deconv3_1'] = self.make_bias(128)
        self.biases['deconv2_1'] = self.make_bias(64)
        self.biases['deconv1_1'] = self.make_bias(image_channels)

    def build(self, images, targets=None):
        self._x = images
        self._t = targets

        # Build the encoder
        self.conv1_1 = self.conv_layer(self._x, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        # Inner-most latent representation
        self._z = self.pool5

        # Build the decoder
        self.deconv5_1 = self.deconv_layer(self._z, self.pool4.shape.as_list(), "deconv5_1")
        self.deconv4_1 = self.deconv_layer(self.deconv5_1, self.pool3.shape.as_list(), "deconv4_1")
        self.deconv3_1 = self.deconv_layer(self.deconv4_1, self.pool2.shape.as_list(), "deconv3_1")
        self.deconv2_1 = self.deconv_layer(self.deconv3_1, self.pool1.shape.as_list(), "deconv2_1")
        self.deconv1_1 = self.deconv_layer(self.deconv2_1, self._x.shape.as_list(), "deconv1_1")

        self._y = self.deconv1_1

        if self._t is not None:
            # Euclidean loss
            self.loss = tf.reduce_sum(tf.square(self._y - self._t))
        else:
            self.loss = None

        return self

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            kernel = self.get_kernel(name)
            bias = self.get_bias(name)

            conv = tf.nn.conv2d(bottom, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biased = tf.nn.bias_add(conv, bias)
            activation = activation_fn(biased)

            return activation

    def deconv_layer(self, bottom, top_shape, name):
        with tf.variable_scope(name):
            kernel = self.get_kernel(name)
            bias = self.get_bias(name)

            deconv = tf.nn.conv2d_transpose(bottom, kernel, top_shape, strides=[1, 2, 2, 1], padding='SAME')
            biased = tf.nn.bias_add(deconv, bias)
            activation = activation_fn(biased)

            return activation

    def make_kernel(self, size, n_input, n_output):
        return tf.Variable(
            tf.random_uniform(
                size + [n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))

    def make_bias(self, n):
        return tf.zeros([n])

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

