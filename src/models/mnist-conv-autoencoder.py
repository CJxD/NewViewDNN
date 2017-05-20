from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import sys, os, os.path

from autoencoder import ConvAutoencoder
from utils import *

usage = "Usage: 32x32-conv-autoencoder.py train/validate <path to inputs> <path to targets>\n"
usage +="       or\n"
usage +="       32x32-conv-autoencoder.py run <path to inputs> <path to outputs>"

# Data parameters
batch_size = 256
shuffle = True
dtype=tf.float32

input_h = input_w = 28
input_ch = 1

# Training parameters
model_file = 'checkpoints/model.ckpt'
learning_rate = 0.01
n_epochs = 1
display_step = 1
examples_to_show = 10

# Network Parameters
filter_sizes = [3, 3, 3]
n_filters = [10, 10, 10]

def batch(images):
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    if shuffle:
        return tf.train.shuffle_batch([images],
            batch_size=batch_size,
            enqueue_many=True,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    else:
        return tf.train.batch([images],
            batch_size=batch_size,
            enqueue_many=True,
            capacity=capacity)

def main(args):
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    global n_epochs, batch_size, shuffle

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)

    train = mnist.train.images
    examples = tf.reshape(train, [-1, input_h, input_w, input_ch])
    batches = batch(examples)
    n_batches = mnist.train.num_examples * n_epochs // batch_size

    net = ConvAutoencoder(filter_sizes, n_filters, input_ch)
    net.build(batches, batches)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(net.loss)

    # Initialize session and graph
    with tf.Session() as sess:
        print("Initialising session")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Main loop
        start_time = time.time()
        for batch_i in range(n_batches):
            batch_time = time.time()

            # Run training
            print("Training batch %d/%d" % (batch_i + 1, n_batches))
            _, loss = sess.run([optimizer, net.loss])
            print("Loss per patch:", loss // batch_size)

            batch_duration = time.time() - batch_time
            elapsed = time.time() - start_time
            print("Took %.3fs, %s elapsed so far" % (batch_duration, time_taken(elapsed)))

        elapsed = time.time() - start_time
        print("Finished in", time_taken(elapsed))

        coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        # Results
        test, _ = mnist.test.next_batch(examples_to_show)
        test_norm = np.array([img - mean_img for img in test])
        examples = tf.reshape(test, [-1, input_h, input_w, input_ch])
        net.build(examples, None)
        recon = sess.run(net.output)
        fig, axs = plt.subplots(2, examples_to_show, figsize=(10, 2))
        for example_i in range(examples_to_show):
            axs[0][example_i].imshow(
                np.reshape(test[example_i, :], (input_w, input_h)))
            axs[1][example_i].imshow(
                np.reshape(
                    np.reshape(recon[example_i, ...], (input_h * input_w,)),
                    (input_h, input_w)))
        fig.show()
        plt.draw()
        plt.waitforbuttonpress()

if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
