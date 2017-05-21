from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import time
import sys, os, os.path

from vgg16_autoencoder import VGG16Autoencoder
from utils import *

usage = "Usage: 32x32-vgg-autoencoder.py train/validate <path to inputs> <path to targets> [path to pretrained model]\n"
usage +="       or\n"
usage +="       32x32-vgg-autoencoder.py run <path to inputs> <path to outputs>"

# Data parameters
batch_size = 256
shuffle = True
input_dtype=tf.uint8
dtype=tf.float32

input_h = input_w = 1024
input_ch = 3
patch_h = patch_w = 32
image_patch_ratio = patch_h * patch_w / (input_h * input_w)
input_noise = 0

# Training parameters
model_file = 'checkpoints/model.ckpt'
learning_rate = 0.001
n_epochs = 1
display_step = 1
examples_to_show = 10

# Network Parameters
filter_sizes = [3, 3, 3]
n_filters = [n * input_ch for n in [10, 10, 10]]

def read_files(image_list):
    filename_queue = tf.train.string_input_producer(image_list, num_epochs=n_epochs, shuffle=shuffle)

    reader = tf.WholeFileReader()
    _, image_file = reader.read(filename_queue)
    image = tf.image.decode_png(image_file, channels=input_ch, dtype=input_dtype)
    image = tf.image.convert_image_dtype(image, dtype)
    image.set_shape((input_h, input_w, input_ch))

    return image

def add_noise(image, mean=0.0, stddev=0.5):
    noise = tf.random_normal(shape=image.shape,
              mean=0.0, stddev=stddev,
              dtype=dtype)

    return image + noise

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

def prepare_batches(image_list, noise=0):
    if noise > 0:
        noise_fn = lambda x: add_noise(x, stddev=noise)
    else:
        noise_fn = lambda x: x

    return batch(generate_patches(noise_fn(read_files(image_list))))

def generate_patches(image):
    '''Splits an image into patches of size patch_h x patch_w
    Input: image of shape [input_h, input_w, input_ch]
    Output: batch of patches shape [n, patch_h, patch_w, input_ch]
    '''
    pad = [[0, 0], [0, 0]]
    patch_area = patch_h * patch_w

    patches = tf.space_to_batch_nd([image], [patch_h, patch_w], pad)
    patches = tf.split(patches, patch_area, 0)
    patches = tf.stack(patches, 3)
    patches = tf.reshape(patches, [-1, patch_h, patch_w, input_ch])

    return patches

def reconstruct_patches(patches):
    '''Reconstructs an image from patches of size patch_h x patch_w
    Input: batch of patches shape [n, patch_h, patch_w, input_ch]
    Output: image of shape [input_h, input_w, input_ch]
    '''
    pad = [[0, 0], [0, 0]]
    patch_area = patch_h * patch_w
    height_ratio = input_h // patch_h
    width_ratio = input_w // patch_w

    image = tf.reshape(patches, [1, height_ratio, width_ratio, patch_area, input_ch])
    image = tf.split(image, patch_area, 3)
    image = tf.stack(image, 0)
    image = tf.reshape(image, [patch_area, height_ratio, width_ratio, input_ch])
    image = tf.batch_to_space_nd(image, [patch_h, patch_w], pad)

    return image[0]

def make_image(data):
    converted = tf.image.convert_image_dtype(data, input_dtype)
    encoded = tf.image.encode_png(converted)

    return encoded

def make_images(data):
    data_queue = tf.train.batch([data],
            batch_size=1,
            enqueue_many=True,
            capacity=10000)

    return make_image(data_queue[0])

def main(args):
    global n_epochs, batch_size, shuffle

    if len(args) < 3 or len(args) > 4:
        print(usage)
        sys.exit(1)

    mode = args[0].lower()
    input_list = args[1]
    output_dir = target_list = args[2]
    if len(args) == 4:
        pretrained_path = args[3]
    else:
        pretrained_path = None

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
    patches_per_img = int(1 // image_patch_ratio)

    if mode == 'run':
        batch_size = patches_per_img
    
    n_patches = n_examples * n_epochs // image_patch_ratio
    n_batches = n_patches // batch_size
    if mode == 'train':
        input_batches = prepare_batches(inputs, noise=input_noise)
    else:
        input_batches = prepare_batches(inputs)

    if mode in ('train', 'validate'):
        with open(target_list, 'r') as target_set:
            targets = target_set.read().splitlines()

        target_batches = prepare_batches(targets)
    else:
        target_batches = None

    net = VGG16Autoencoder(input_ch, pretrained_path)
    net.build(input_batches, target_batches)

    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(net.loss)
    elif mode == 'run':
        input_data = reconstruct_patches(net.input)
        output_data = reconstruct_patches(net.output)
        input_image = make_image(input_data)
        patch_images = make_images(net.output)
        output_image = make_image(output_data)

    saver = tf.train.Saver()

    # Initialize session and graph
    with tf.Session() as sess:
        if os.path.isfile(model_file + '.index'):
            print("Using model from", model_file)

            try:
                saver.restore(sess, model_file)
            except tf.errors.OpError:
                # Incompatible model
                print("Could not load model - initialising new session")
                sess.run(tf.global_variables_initializer())
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
                    print("Training batch %d/%d" % (i, n_batches))
                    _, loss = sess.run([optimizer, net.loss])
                    patch_loss = loss // batch_size
                    print("Loss per patch: %d (%.2f%%)" % (patch_loss, 100 * patch_loss / (patch_h * patch_w * input_ch)))

                # Validate
                elif mode == 'validate':
                    print("Validating batch %d/%d" % (i, n_batches))
                    loss.append(sess.run([net.loss]))

                # Generate outputs
                elif mode == 'run':
                    print("Processing image %d/%d" % (i, n_batches))
                    tag = str(i)
                    fname_in = tf.constant(os.path.join(output_dir, tag + '_in.png'))
                    fname_out = tf.constant(os.path.join(output_dir, tag + '_out.png'))
                    fwrite_in = tf.write_file(fname_in, input_image)
                    fwrite_out = tf.write_file(fname_out, output_image)
                    sess.run([fwrite_in, fwrite_out])

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
            if not os.path.exists(directory):
                os.makedirs(directory)
            save_path = saver.save(sess, model_file)
            print("Model saved to", save_path)

        # Results
        if mode == 'train':
            print("Final image loss:", loss / image_patch_ratio // batch_size)
        
        elif mode == 'validate':
            print("Average image loss:", np.mean(loss) / image_patch_ratio // batch_size)

if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
