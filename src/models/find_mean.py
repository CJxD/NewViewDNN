#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import *
from utils import generate_patches

usage = "Usage: find_mean.py <TFRecords dataset path>"

input_dtype = tf.uint8
dtype = tf.float32
input_h = input_w = 1024
input_ch = 3
patch_h = patch_w = 32
num_samples = 100

def read_records(record_list):
    filename_queue = tf.train.string_input_producer(record_list, num_epochs=1)

    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)

    features = tf.parse_single_example(
        example,
        features={
            'collection': tf.FixedLenFeature([], tf.string),
            'model': tf.FixedLenFeature([], tf.string),
            'angle': tf.FixedLenFeature([], tf.float32),
            'input_image':  tf.FixedLenFeature([], tf.string),            
            'target_image': tf.FixedLenFeature([], tf.string, default_value=''),
        })

    input_image = decode_image(features['input_image'])
    target_image = decode_image(features['target_image'])

    return input_image, target_image

def decode_image(encoded):
    with tf.variable_scope('decode_image'):
        image = tf.image.decode_png(encoded, channels=input_ch, dtype=input_dtype)
        image = tf.image.convert_image_dtype(image, dtype)
        image = tf.image.resize_images(image, [input_h, input_w])

    return image

def encode_image(data):
    with tf.variable_scope('encode_image'):
        converted = tf.image.convert_image_dtype(data, input_dtype)
        encoded = tf.image.encode_png(converted)

    return encoded

def mix(a, b, alpha):
    return a * tf.constant(1-alpha) + b * tf.constant(alpha)

def main(args):
    if len(args) < 1:
        print(usage, file=sys.stderr)
        sys.exit(1)

    input_image, _ = read_records([args[0]])
    input_patches = generate_patches(input_image, patch_h, patch_w)

    mean = tf.Variable(tf.zeros([patch_h, patch_w, input_ch], dtype))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        hue_list = []
        for i in range(num_samples):
            alpha = 1.0 / (i+1)

            batch_mean = tf.reduce_mean(input_patches, axis=0)
            update_mean = mean.assign(mix(mean, batch_mean, alpha))

            hsv = tf.image.rgb_to_hsv(input_patches)

            hues, _ = sess.run([hsv, update_mean])
            for image in hues:
                for y in image:
                    for x in y:
                        h, s, l = x
                        hue_list.append(h)

            print("Processed image %d/%d" % (i, num_samples))

        mean_image = encode_image(mean)

        output_dir = dirname(args[0])
        output_name = splitext(basename(args[0]))[0]
        
        hist, bins = np.histogram(hue_list, bins=64)
        np.save(join(output_dir, output_name + '-histogram.npy'), (hist, bins))

        fname = tf.constant(join(output_dir, output_name + '-mean.png'))
        fwrite = tf.write_file(fname, mean_image)

        sess.run([fwrite])
        coord.request_stop()
        coord.join(threads)

    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.ylim([0, np.percentile(hist, 90)])
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
