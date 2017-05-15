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
batch_size = 1024
shuffle = True
input_dtype=tf.uint8
dtype=tf.float32

input_h = input_w = 1024
input_ch = 3
patch_h = patch_w = 32
image_patch_ratio = patch_h * patch_w / (input_h * input_w)

# Training parameters
model_file = 'checkpoints/model.ckpt'
learning_rate = 0.01
n_epochs = 10
display_step = 1
examples_to_show = 10

# Network Parameters
filter_sizes = [3, 3, 3, 3]
n_filters = [1, 10, 10, 10]

def read_files(image_list):
    filename_queue = tf.train.string_input_producer(image_list, num_epochs=n_epochs)

    reader = tf.WholeFileReader()
    _, image_file = reader.read(filename_queue)
    image = tf.image.decode_png(image_file, channels=input_ch, dtype=input_dtype)
    image = tf.image.convert_image_dtype(image, dtype)
    image.set_shape((input_h, input_w, input_ch))

    return image

def generate_patches(image):
    patch_size = [1, patch_h, patch_w, 1]
    patches = tf.extract_image_patches([image],
        patch_size, patch_size, [1, 1, 1, 1], 'VALID')
    patches = tf.reshape(patches, [-1, patch_h, patch_w, input_ch])

    return patches

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

def prepare_batches(image_list):
    return batch(generate_patches(read_files(image_list)))

def reconstruct_image(patches):
    image = tf.reshape(patches, [1, input_h, input_w, input_ch])
    converted = tf.image.convert_image_dtype(image[0], input_dtype)
    encoded = tf.image.encode_png(converted)

    return encoded

def main(args):
    global n_epochs, shuffle

    if len(args) != 3:
        print(usage)
        sys.exit(1)

    mode = args[0].lower()
    input_list = args[1]
    output_dir = target_list = args[2]

    if mode not in ('train', 'validate', 'run'):
        print(usage)
        sys.exit(1)

    if mode == 'validate':
        loss = []

    if mode in ('validate', 'run'):
        n_epochs = 1
        shuffle = False

    with open(input_list, 'r') as input_set:
        inputs = input_set.read().splitlines()

    n_examples = len(inputs)
    n_patches = n_examples * n_epochs // image_patch_ratio
    n_batches = n_patches // batch_size
    input_batches = prepare_batches(inputs)

    patches_per_img = int(1 // image_patch_ratio)
    batches_per_img = patches_per_img / batch_size
    capacity = patches_per_img * 3
    output_buf = tf.FIFOQueue(capacity,
                     dtypes=[dtype],
                     shapes=[[patch_h, patch_w, input_ch]])

    if mode in ('train', 'validate'):
        with open(target_list, 'r') as target_set:
            targets = target_set.read().splitlines()

        target_batches = prepare_batches(targets)
    else:
        target_batches = None

    net = ConvAutoencoder(n_filters, filter_sizes)
    net.prepare(input_batches, target_batches)

    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(net.loss)

    saver = tf.train.Saver()

    # Initialize session and graph
    with tf.Session() as sess:
        if os.path.isfile(model_file + '.index'):
            saver.restore(sess, model_file)
            print("Using model from", model_file)
        else:
            if mode in ('validate', 'run'):
                print("No trained model found in %s" % model_file, file=sys.stderr)
                sys.exit(1)
            else:
                print("Initialising session")
                sess.run(tf.global_variables_initializer())
            
        sess.run(tf.local_variables_initializer())

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Main loop
        try:
            i = 0
            start_time = time.time()
            while not coord.should_stop():
                batch_time = time.time()

                # Train
                if mode == 'train':
                    # Run training steps or whatever
                    print("Training batch %d/%d" % (i, n_batches))
                    _, loss = sess.run([optimizer, net.loss])
                    print("Loss per patch:", loss // batch_size)

                elif mode == 'validate':
                    print("Validating batch %d/%d" % (i, n_batches))
                    loss.append(sess.run([net.loss]))

                elif mode == 'run':
                    print("Processing batch %d/%d" % (i, n_batches))
                    output = sess.run([net.y])
                    
                    enqueue = output_buf.enqueue_many(output)
                    dequeue = output_buf.dequeue_many(patches_per_img)
                    image = reconstruct_image(dequeue)

                    if (i - 1) % batches_per_img == 0:
                        tag = str(i // batches_per_img)
                        fname = tf.constant(os.path.join(output_dir, tag + '.png'))
                        fwrite = tf.write_file(fname, image)
                        sess.run([fwrite])
                    else:
                        sess.run([enqueue])

                batch_duration = time.time() - batch_time
                elapsed = time.time() - start_time
                print("Took %.3fs, %s elapsed so far" % (batch_duration, time_taken(elapsed)))

                i += 1

        except tf.errors.OutOfRangeError:
            elapsed = time.time() - start_time
            print("Finished in", time_taken(elapsed))
        finally:
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        # Save model
        if mode == 'train':
            directory = os.path.dirname(model_file)
            os.makedirs(directory)
            save_path = saver.save(sess, model_file)
            print("Model saved to", save_path)

        # Results
        if mode == 'train':
            print("Final image loss:", loss / image_patch_ratio // batch_size)
        
        elif mode == 'validate':
            print("Average image loss:", np.mean(loss) / image_patch_ratio // batch_size)

        elif mode == 'run':
            # Plot example reconstructions
            test_xs, _ = mnist.test.next_batch(examples_to_show)
            test_xs_norm = np.array([img - mean_img for img in test_xs])
            recon = sess.run(net.y, feed_dict={net.x: test_xs_norm})
            print(recon.shape)
            fig, axs = plt.subplots(2, examples_to_show, figsize=(10, 2))
            for example_i in range(examples_to_show):
                axs[0][example_i].imshow(
                    np.reshape(test_xs[example_i, :], (input_h, input_w)))
                axs[1][example_i].imshow(
                    np.reshape(
                        np.reshape(recon[example_i, ...], (input_h * input_w,)) + mean_img,
                        (input_h, input_w)))
            fig.show()
            plt.draw()
            plt.waitforbuttonpress()

if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
