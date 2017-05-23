from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import time
import sys, os, os.path

from autoencoder import ConvAutoencoder
from utils import *

script_name = os.path.basename(__file__)

usage = "Usage: %s train/validate <path to examples>\n" % script_name
usage +="       or\n"
usage +="       %s run <path to inputs> <path to outputs>" % script_name

# Data parameters
batch_size = 256
shuffle = False
input_dtype=tf.uint8
dtype=tf.float32

input_h = input_w = 1024
input_ch = 3
patch_h = patch_w = 32
image_patch_ratio = patch_h * patch_w / (input_h * input_w)
patches_per_img = int(1 // image_patch_ratio)

# Training parameters
model_file = 'checkpoints/model.ckpt'
log_dir = 'logs'
learning_rate = 0.001
n_epochs = 1

summary_interval = 10
checkpoint_interval = 500

# Network Parameters
filter_sizes = [3, 3, 3]
n_filters = [n * input_ch for n in [10, 10, 10]]

def read_files(image_list):
    filename_queue = tf.train.string_input_producer(image_list, num_epochs=n_epochs)

    reader = tf.WholeFileReader()
    _, image_file = reader.read(filename_queue)

    return decode_image(image_file)

def read_records(record_list):
    filename_queue = tf.train.string_input_producer(record_list, num_epochs=n_epochs)

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
    image = tf.image.decode_png(encoded, channels=input_ch, dtype=input_dtype)
    image = tf.image.convert_image_dtype(image, dtype)
    image = tf.image.resize_images(image, [input_h, input_w])

    return image

def encode_image(data):
    converted = tf.image.convert_image_dtype(data, input_dtype)
    encoded = tf.image.encode_png(converted)

    return encoded

def encode_images(data):
    data_queue = tf.train.batch([data],
            batch_size=1,
            enqueue_many=True,
            capacity=10000)

    return encode_image(data_queue[0])

def batch(tensors):
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    if shuffle:
        return tf.train.shuffle_batch(tensors,
            batch_size=batch_size,
            enqueue_many=True,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    else:
        return tf.train.batch(tensors,
            batch_size=batch_size,
            enqueue_many=True,
            capacity=capacity,
            num_threads=1)

def main(args):
    global n_epochs, batch_size, shuffle

    # Parse arguments
    try:
        mode = args[0].lower()

        if mode in ('train', 'validate'):
            input_path = args[1]
        elif mode in ('run'):
            input_path = args[1]
            output_dir = args[2]
        else:
            raise ValueError("Mode must be one of train/validate/run")
    except:
        print(usage, file=sys.stderr)
        sys.exit(1)

    # Global variable adjustments
    if mode in ('validate', 'run'):
        n_epochs = 1
        shuffle = False
        batch_size = patches_per_img

    # Data loading
    if os.path.splitext(input_path)[1] == '.tfrecords':
        input_images, target_images = read_records([input_path])

        input_patches = generate_patches(input_images, patch_h, patch_w)
        target_patches = generate_patches(target_images, patch_h, patch_w)

        input_batches, target_batches = batch([input_patches, target_patches])

        n_examples = 0
        for record in tf.python_io.tf_record_iterator(input_path):
            n_examples += 1
    else:
        with open(input_path) as input_file:
            input_list = input_file.read().splitlines()

        input_images = read_files(input_list)
        input_patches = generate_patches(input_images, patch_h, patch_w)
        input_batches = batch([input_patches])

        target_batches = None
        
        n_examples = len(input_list)

    if mode in ('train', 'validate'):
        if target_batches is None:
            print("No targets specified, use TFRecords for training data.", file=sys.stderr)
            sys.exit(1)

    # Stats
    n_patches = n_examples * n_epochs * patches_per_img
    n_batches = n_patches // batch_size

    # Network
    net = ConvAutoencoder(filter_sizes, n_filters, input_ch)
    net.build(input_batches, target_batches)

    # Network outputs
    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(net.loss)
    
    if mode == 'validate':
        losses = []
    
    if mode in ('validate', 'run'):
        input_data = reconstruct_image(net.input, input_h, input_w)
        output_data = reconstruct_image(net.output, input_h, input_w)
        input_image = encode_image(input_data)
        patch_images = encode_images(net.output)
        output_image = encode_image(output_data)

    if summary_interval > 0:
         # Summaries
         tf.summary.scalar("loss", net.loss)

         if mode == 'validate':
             tf.summary.image("input", input_data)
             tf.summary.image("output", output_data)

    # Initialize session and graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Setup logging
        log_path = os.path.join(log_dir, mode)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        summary = tf.summary.merge_all()
        log_writer = tf.summary.FileWriter(log_path)
        log_writer.add_graph(sess.graph)

        # Restore model
        if os.path.isfile(model_file + '.index'):
            print("Using model from", model_file)

            try:
                saver.restore(sess, model_file)
            except tf.errors.OpError:
                # Incompatible model
                if mode == 'train':
                    print("Could not load model - initialising new session")
                    sess.run(tf.global_variables_initializer())
                else:
                    print("No trained model found in %s" % model_file, file=sys.stderr)
                    sys.exit(1)
        else:
            if mode == 'train':
                print("Initialising session")
                sess.run(tf.global_variables_initializer())
            else:
                print("No trained model found in %s" % model_file, file=sys.stderr)
                sys.exit(1)
            
        sess.run(tf.local_variables_initializer())

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Main loop
        try:
            step = 0
            start_time = time.time()
            while not coord.should_stop():
                batch_time = time.time()

                # Write summary
                if summary_interval > 0 and step % summary_interval == 0:
                    s = sess.run(summary)
                    log_writer.add_summary(s, step)

                # Train
                if mode == 'train':
                    print("Training batch %d/%d" % (step, n_batches))
                    _, loss = sess.run([optimizer, net.loss])
                    patch_loss = loss // batch_size
                    print("Loss per patch: %d (%.2f%%)" % (patch_loss, 100 * patch_loss / (patch_h * patch_w * input_ch)))

                # Validate
                elif mode == 'validate':
                    print("Validating batch %d/%d" % (step, n_batches))
                    losses.append(sess.run(net.loss))

                # Generate outputs
                elif mode == 'run':
                    print("Processing image %d/%d" % (step, n_batches))
                    tag = str(step)
                    fname_in = tf.constant(os.path.join(output_dir, tag + '_in.png'))
                    fname_out = tf.constant(os.path.join(output_dir, tag + '_out.png'))
                    fwrite_in = tf.write_file(fname_in, input_image)
                    fwrite_out = tf.write_file(fname_out, output_image)
                    sess.run([fwrite_in, fwrite_out])

                # Write checkpoint
                if step % checkpoint_interval == 0:
                    saver.save(sess, model_file, step)

                batch_duration = time.time() - batch_time
                elapsed = time.time() - start_time
                print("Took %.3fs, %s elapsed so far" % (batch_duration, time_taken(elapsed)))

                step += 1

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
            print("Average image loss:", np.mean(losses) / image_patch_ratio // batch_size)

if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
