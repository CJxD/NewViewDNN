#!/usr/bin/env python3

import tensorflow as tf
import sys
from os.path import *

usage = "Usage: find_mean.py <TFRecords dataset path>"

input_dtype = tf.uint8
dtype = tf.float32
input_h = input_w = 1024
input_ch = 4
batch_size = 500

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

    input_image, target_image = read_records([args[0]])
    num_examples = sum(1 for _ in tf.python_io.tf_record_iterator(args[0]))

    input_mean = tf.Variable(tf.zeros(input_image.shape, dtype))
    target_mean = tf.Variable(tf.zeros(target_image.shape, dtype))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(num_examples):
            alpha = 1.0 / (i+1)

            update_input_mean = input_mean.assign(mix(input_mean, input_image, alpha))
            update_target_mean = target_mean.assign(mix(input_mean, input_image, alpha))

            sess.run([update_input_mean, update_target_mean])

            print("Processed image %d/%d" % (i, num_examples))

        input_mean_image = encode_image(input_mean)
        target_mean_image = encode_image(target_mean)

        output_dir = dirname(args[0])
        output_name = splitext(args[0])[0]
        
        fname_in = tf.constant(join(output_dir, output_name + '-input-mean.png'))
        fname_tgt = tf.constant(join(output_dir, output_name + '-target-mean.png'))
        fwrite_in = tf.write_file(fname_in, input_mean_image)
        fwrite_tgt = tf.write_file(fname_tgt, target_mean_image)

        sess.run([fwrite_in, fwrite_tgt])
        coord.request_stop()
        coord.join(threads) 

if __name__ == '__main__':
    main(sys.argv[1:])
