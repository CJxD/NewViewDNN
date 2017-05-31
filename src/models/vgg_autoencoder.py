from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import math

from networks import CNN
from utils import lrelu

class VGG16Autoencoder(CNN):
    def __init__(self, image_channels=1, pretrained_path=None):
        super().__init__()
        self.deconv_strides = [1, 2, 2, 1]

        if pretrained_path is not None:
            print("Using pretrained weights from", pretrained_path)
            self.load_weights(image_channels, pretrained_path)
        else:
            self.init_weights(image_channels)

    def load_weights(self, image_channels, path):
        if image_channels != 3:
            raise ValueError("Input must have 3 channels to use pretrained VGG weights")

        self.data = np.load(path, encoding='latin1').item()
        self.init_deconv(image_channels)

    def init_weights(self, image_channels):
        self.init_conv(image_channels)
        self.init_deconv(image_channels)

    def init_conv(self, image_channels):
        self.data['conv1_1'] = [
            self.make_kernel([3, 3], image_channels, 64),
            self.make_bias(64)
        ]
        self.data['conv1_2'] = [
            self.make_kernel([3, 3], 64, 64),
            self.make_bias(64)
        ]

        self.data['conv2_1'] = [
            self.make_kernel([3, 3], 64, 128),
            self.make_bias(128)
        ]
        self.data['conv2_2'] = [
            self.make_kernel([3, 3], 128, 128),
            self.make_bias(128)
        ]

        self.data['conv3_1'] = [
            self.make_kernel([3, 3], 128, 256),
            self.make_bias(256)
        ]
        self.data['conv3_2'] = [
            self.make_kernel([3, 3], 256, 256),
            self.make_bias(256)
        ]
        self.data['conv3_3'] = [
            self.make_kernel([3, 3], 256, 256),
            self.make_bias(256)
        ]

        self.data['conv4_1'] = [
            self.make_kernel([3, 3], 256, 512),
            self.make_bias(512)
        ]
        self.data['conv4_2'] = [
            self.make_kernel([3, 3], 512, 512),
            self.make_bias(512)
        ]
        self.data['conv4_3'] = [
            self.make_kernel([3, 3], 512, 512),
            self.make_bias(512)
        ]

        self.data['conv5_1'] = [
            self.make_kernel([3, 3], 512, 512),
            self.make_bias(512)
        ]
        self.data['conv5_2'] = [
            self.make_kernel([3, 3], 512, 512),
            self.make_bias(512)
        ]
        self.data['conv5_3'] = [
            self.make_kernel([3, 3], 512, 512),
            self.make_bias(512)
        ]

    def init_deconv(self, image_channels):
        self.data['deconv5_1'] = [
            self.make_kernel([4, 4], 512, 512),
            self.make_bias(512)
        ]
        self.data['deconv4_1'] = [
            self.make_kernel([4, 4], 256, 512),
            self.make_bias(256)
        ]
        self.data['deconv3_1'] = [
            self.make_kernel([4, 4], 128, 256),
            self.make_bias(128)
        ]
        self.data['deconv2_1'] = [
            self.make_kernel([4, 4], 64, 128),
            self.make_bias(64)
        ]
        self.data['deconv1_1'] = [
            self.make_kernel([4, 4], image_channels, 64),
            self.make_bias(image_channels)
        ]

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

        return self
