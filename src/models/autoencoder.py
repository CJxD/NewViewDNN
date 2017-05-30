from __future__ import division, print_function
import tensorflow as tf

from networks import CNN
from utils import lrelu

class ConvAutoencoder(CNN):
    def __init__(self, filter_sizes=[3, 3, 3], n_filters=[10, 10, 10], image_channels=1):
        super().__init__()
        self.activation = tf.nn.relu
        self.conv_strides = [1, 1, 1, 1]
        self.deconv_strides = [1, 2, 2, 1]

        self.n_filters = [image_channels] + n_filters
        self.filter_sizes = filter_sizes

        # Initialise weights
        for i, n_output in enumerate(self.n_filters[1:]):
            n_input = self.n_filters[i]
            kernel_size = [self.filter_sizes[i], self.filter_sizes[i]]
            
            id = str(i+1)
            # Convolution weights
            self.data['conv' + id] = [
                self.make_kernel(kernel_size, n_input, n_output),
                self.make_bias(n_output)
            ]

            # Deconvolution weights
            name = 'deconv' + id
            self.data['deconv' + id] = [
                self.make_kernel(kernel_size, n_input, n_output),
                self.make_bias(n_input)
            ]

    def build(self, images, targets=None):
        self._x = images
        self._t = targets
        shapes = []
        prev_layer = self._x

        # Build the encoder
        for i, _ in enumerate(self.n_filters[1:]):
            name = 'conv' + str(i+1)
            shapes.append(prev_layer.shape.as_list())
            prev_layer = self.conv_layer(prev_layer, name)
            prev_layer = self.max_pool(prev_layer)

        # Inner-most latent representation
        self._z = prev_layer

        # Build the decoder
        for i, shape in enumerate(shapes[::-1]):
            name = 'deconv' + str(len(shapes) - i)
            prev_layer = self.deconv_layer(prev_layer, shape, name)

        self._y = prev_layer

        if self._t is not None:
            # Euclidean loss
            # Weight the loss by regions with the highest difference
            mask = self.weighted_diff(self._x, self._t)
            self.loss = self.euclidean_loss((self._y - self._t) * mask)
        else:
            self.loss = None

        return self

