#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import collections
import time
import sys, os, os.path
import argparse

from utils import *

# Defaults
batch_size = -1
shuffle = False
input_dtype=tf.uint8
dtype=tf.float32

learning_rate = 0.001
num_epochs = 1

input_h = input_w = 1024
input_ch = 3
patch_h = patch_w = 32
image_patch_ratio = lambda: patch_h * patch_w / (input_h * input_w)
patches_per_img = lambda: int(1 // image_patch_ratio())

filter_sizes = [3, 3, 3]
num_filters = [n * input_ch for n in [10, 10, 10]]

summary_interval = 10
checkpoint_interval = 500
model_file = 'checkpoints/model.ckpt'
log_dir = 'logs'

def read_files(image_list):
    filename_queue = tf.train.string_input_producer(image_list, num_epochs=num_epochs)

    reader = tf.WholeFileReader()
    _, image_file = reader.read(filename_queue)

    return decode_image(image_file)

def read_records(record_list):
    filename_queue = tf.train.string_input_producer(record_list, num_epochs=num_epochs)

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

def encode_images(data):
    data_queue = tf.train.batch([data],
            batch_size=1,
            enqueue_many=True,
            capacity=10000)

    return encode_image(data_queue[0])

def batch(tensors, batch_size=batch_size):
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
    '''
    Data loading
    '''
    if os.path.splitext(args.input_files[0])[1] == '.tfrecords':
        input_images, target_images = read_records(args.input_files)

        input_patches = generate_patches(input_images, patch_h, patch_w)
        target_patches = generate_patches(target_images, patch_h, patch_w)

        input_batches, target_batches = batch([input_patches, target_patches], batch_size)

        num_examples = 0
        for f in args.input_files:
            for record in tf.python_io.tf_record_iterator(f):
                num_examples += 1
    else:
        input_list = []
        for f in args.input_files:
            with open(f) as input_file:
                input_list += input_file.read().splitlines()

        input_images = read_files(input_list)
        input_patches = generate_patches(input_images, patch_h, patch_w)
        input_batches = batch([input_patches], batch_size)

        target_batches = None
        
        num_examples = len(input_list)

    if args.mode in ('train', 'validate'):
        if target_batches is None:
            print("No targets specified, use TFRecords for training data.", file=sys.stderr)
            sys.exit(1)

    '''
    Stats
    '''
    num_patches = num_examples * num_epochs * patches_per_img()
    num_batches = num_patches // batch_size

    '''
    Network
    '''
    if args.model == 'basic':
        from autoencoder import ConvAutoencoder
        net = ConvAutoencoder(args.filter_sizes, args.num_filters, input_ch)
        
    elif args.model == 'vgg16':
        from vgg_autoencoder import VGG16Autoencoder
        if args.pretrain_weights:
            net = VGG16Autoencoder(input_ch, args.pretrain_weights)
        else:
            net = VGG16Autoencoder(input_ch)
    
    else:
        print("Invalid model: %s.", file=sys.stderr)
        sys.exit(1)
        
    if args.differential and target_batches is not None:
        target_batches = target_batches - input_batches

    net.build(input_batches, target_batches)

    '''
    Network outputs
    '''
    if args.mode == 'train':
        learning_loss = net.weighted_loss(base_weight=0.5, name="learning_loss")
        loss = net.euclidean_loss(name="image_loss")
        losses = []
        optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(learning_loss)
    
    elif args.mode == 'validate':
        loss = net.weighted_loss(base_weight=0.5, name="image_loss")
        losses = []

    elif args.mode == 'run':
        if net.target is not None:
            loss = net.euclidean_loss(name="image_loss")
        else:
            loss = tf.constant(0)

    input = net.input
    output = net.output
    target = net.target

    if args.differential:
        output += net.input
        if target is not None:
            target += net.input

    if args.mode == 'train':
        input, output, target = batch([input, output, target], patches_per_img())

    input_data = reconstruct_image(input, input_h, input_w)
    output_data = reconstruct_image(output, input_h, input_w)
    input_image = encode_image(input_data)
    output_image = encode_image(output_data)
    patch_images = encode_images(net.output)
    if target is not None:
        target_data = reconstruct_image(target, input_h, input_w)
        target_image = encode_image(target_data)

    if args.summary_interval > 0:
         # Summaries
         tf.summary.scalar("loss", loss)

         tf.summary.image("input", [input_data])
         tf.summary.image("output", [output_data])
         if target is not None:
             tf.summary.image("target", [target_data])

    '''
    Initialize session and graph
    '''
    saver = tf.train.Saver()
    with tf.Session() as sess:
        '''
        Setup logging
        '''
        summary = tf.summary.merge_all()
        log_writer = tf.summary.FileWriter(os.path.join(args.log_dir, args.mode))
        log_writer.add_graph(sess.graph)

        '''
        Restore model
        '''
        if os.path.isfile(args.model_file + '.index'):
            print("Using model from", args.model_file)

            try:
                saver.restore(sess, args.model_file)
            except tf.errors.OpError:
                # Incompatible model
                if args.mode == 'train':
                    print("Could not load model - initialising new session")
                    sess.run(tf.global_variables_initializer())
                else:
                    print("No trained model found in %s" % args.model_file, file=sys.stderr)
                    sys.exit(1)
        else:
            if args.mode == 'train':
                print("Initialising session")
                sess.run(tf.global_variables_initializer())
            else:
                print("No trained model found in %s" % args.model_file, file=sys.stderr)
                sys.exit(1)
            
        sess.run(tf.local_variables_initializer())

        '''
        Start input enqueue threads
        '''
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        '''
        Main loop
        '''
        try:
            step = 0
            start_time = time.time()
            remaining = collections.deque(maxlen=100)
            while not coord.should_stop():
                batch_time = time.time()

                # Prepare summary if required
                if args.summary_interval > 0 and step % args.summary_interval == 0:
                    summary_runner = summary
                else:
                    summary_runner = tf.constant(0)

                # Train
                if args.mode == 'train':
                    print("Training batch %d/%d" % (step, num_batches))
                    _, l, s = sess.run([optimizer, loss, summary])
                    patch_loss = l / batch_size
                    print("Loss per patch: %.1f (%.2f%%)" % (patch_loss, 100 * patch_loss / (patch_h * patch_w * input_ch)))
                    losses.append(l)

                # Validate
                elif args.mode == 'validate':
                    print("Validating batch %d/%d" % (step, num_batches))
                    loss = tf.reduce_sum(tf.square(target_batches - input_batches))
                    summary = tf.constant(0)
                    l, s = sess.run([loss, summary])
                    losses.append(l)

                # Generate outputs
                elif args.mode == 'run':
                    print("Processing image %d/%d" % (step, num_batches))
                    tag = str(step)
                    fname_in = tf.constant(os.path.join(args.output_dir, tag + '_in.png'))
                    fname_out = tf.constant(os.path.join(args.output_dir, tag + '_out.png'))
                    fwrite_in = tf.write_file(fname_in, input_image)
                    fwrite_out = tf.write_file(fname_out, output_image)

                    if target is not None:
                        fname_tgt = tf.constant(os.path.join(args.output_dir, tag + '_tgt.png'))
                        fwrite_tgt = tf.write_file(fname_tgt, target_image)
                    else:
                        fwrite_tgt = tf.constant(0)

                    _, _, _, s = sess.run([fwrite_in, fwrite_out, fwrite_tgt, summary])

                # Write summary
                if s:
                    log_writer.add_summary(s, step)

                # Write checkpoint
                if step % args.checkpoint_interval == 0:
                    saver.save(sess, args.model_file, step)

                step += 1

                batch_duration = time.time() - batch_time
                elapsed = time.time() - start_time
                if step > 10:
                    remaining.append(batch_duration * (num_batches - step))
                    avg_remaining = np.mean(remaining)
                    remaining_text = ", %s remaining" % time_taken(avg_remaining)
                else:
                    remaining_text = ""

                print("Took %.3fs, %s elapsed so far%s" % (batch_duration, time_taken(elapsed), remaining_text))

        except tf.errors.OutOfRangeError:
            elapsed = time.time() - start_time
            print("Finished in", time_taken(elapsed))
        finally:
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        '''
        Save model
        '''
        if args.mode == 'train':
            save_path = saver.save(sess, args.model_file)
            print("Model saved to", save_path)

        '''
        Results
        '''
        if args.mode in ('train', 'validate'):
            loss_per_patch = np.median(losses) / batch_size
            loss_per_image = loss_per_patch / image_patch_ratio()
            print("Average patch loss:", loss_per_patch)
            print("Average image loss:", loss_per_image)
            print("Standard deviation:", np.std(losses))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convolutional Autoencoder Runner')
    parser.add_argument('mode', help="train, validate, or run")
    
    parser.add_argument('-i', '--input-files', nargs='+', required=True, help="Input TFRecords files or list of input images (run mode only) [required]")
    parser.add_argument('-o', '--output-dir', help="Directory to store output images [required in run mode]")

    parser.add_argument('-b', '--batch-size', default=batch_size, type=int, help="Number of examples per batch (default: num patches per image)")
    parser.add_argument('-s', '--shuffle', action='store_true', help="Shuffles input batches before training")
    parser.add_argument('-m', '--model', default='basic', help="basic (default) or vgg16")
    parser.add_argument('-w', '--pretrain-weights', help="Pretrained weights for the model, if applicable")
    parser.add_argument('-d', '--differential', action='store_true', help="Train for the differences between input and output images rather than the output image itself.")
    
    parser.add_argument('--filter-sizes', nargs='+', default=filter_sizes, type=int, help="List of kernel filter sizes to use for each convolutional layer (basic model only)")
    parser.add_argument('--num-filters', nargs='+', default=num_filters, type=int, help="Output depth for each convolutional layer (basic model only)")
 
    parser.add_argument('-l', '--learning-rate', default=learning_rate, type=float, help="Initial learning rate (default: %f)" % learning_rate)
    parser.add_argument('-n', '--num-epochs', default=num_epochs, type=int, help="Number of training repetitions to do (default: %d)" % num_epochs)
    
    parser.add_argument('-X', '--image-width', default=input_w, type=int, help="Resize input images to this width (default: %d)" % input_w)
    parser.add_argument('-Y', '--image-height', default=input_h, type=int, help="Resize input images to this height (default: %d)" % input_h)
    parser.add_argument('-C', '--image-channels', default=input_ch, type=int, help="Use grayscale (1), RGB (3), or RGBA (4) (default: %d)" % input_ch)

    parser.add_argument('-x', '--patch-width', default=patch_w, type=int, help="Width of patches to split input images into (default: %d)" % patch_w)
    parser.add_argument('-y', '--patch-height', default=patch_h, type=int, help="Height of patches to split input images into (default: %d)" % patch_h)

    parser.add_argument('--summary-interval', default=summary_interval, type=int, help="Number of batches before each summary checkpoint (default: %d)" % summary_interval)
    parser.add_argument('--checkpoint-interval', default=checkpoint_interval, type=int, help="Number of batches before each model checkpoint (default: %d)" % checkpoint_interval)
    parser.add_argument('--log-dir', default=log_dir, help="Directory to save summaries to (default: %s)" % log_dir)
    parser.add_argument('--model-file', default=model_file, help="File to save model to (default: %s)" % model_file)
    
    args = parser.parse_args()

    # Globals
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    shuffle = args.shuffle
    input_h = args.image_height
    input_w = args.image_width
    input_ch = args.image_channels
    patch_h = args.patch_height
    patch_w = args.patch_width
    
    # Pre-checks
    if batch_size < 0:
        args.batch_size = batch_size = patches_per_img()

    if args.mode not in ('train', 'validate', 'run'):
        parser.print_help()
        print("Mode must be one of train/validate/run.", file=sys.stdout)
        sys.exit(2)
        
    if args.mode == 'run':
        if not args.output_dir:
            parser.print_help()
            print("Must specify output dir when in run mode.", file=sys.stdout)
            sys.exit(2)
        else:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
    
    if args.mode in ('validate', 'run'):
        num_epochs = 1
        shuffle = False
        batch_size = patches_per_img()

    checkpoint_dir = os.path.dirname(args.model_file)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    path = os.path.join(args.log_dir, args.mode)
    if not os.path.exists(path):
        os.makedirs(path)

    main(args)
