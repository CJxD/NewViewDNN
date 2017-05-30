#!/usr/bin/env python3

import tensorflow as tf
import sys
import re
from os.path import *
from glob import glob

base_path = dirname(dirname(realpath(__file__)))
corpus_path = join(base_path, "data", ".shapenet-nocorrupt")
output_path = join(base_path, "data")
train_file = join(output_path, "train-nc.tfrecords")
val_file = join(output_path, "val-nc.tfrecords")
test_file = join(output_path, "test-nc.tfrecords")

usage = "Usage: create_inputs.py [collection id, ...]"

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def main(args):
	if len(args) >= 1:
		try:
			collections = [int(arg) for arg in args]
		except:
			print(usage, file=sys.stderr)
			return 1
	else:
		collections = None

	proj_pat = re.compile(r'.*/proj_(.+)\.png')
	
	with open(join(corpus_path, "splits.csv"), 'r') as splits, \
		tf.python_io.TFRecordWriter(train_file) as train, \
		tf.python_io.TFRecordWriter(val_file) as val, \
		tf.python_io.TFRecordWriter(test_file) as test:
		header = None
		for line in splits:
			if not header:
				header = line
				continue
				
			entry = line.strip().split(',')
			synset_num = int(entry[1])
			
			if collections is not None and synset_num not in collections:
				continue
			
			synset_id = '0' + str(synset_num)
			id = entry[3]
			split = entry[4]
			dir = join(corpus_path, synset_id, id, "renders")

			if split == "train":
				records = train
			elif split == "val":
				records = val
			elif split == "test":
				records = test
			else:
				continue

			for file in glob(join(dir, "proj_*.png")):
				try:
					matches = proj_pat.match(file)
					angle = matches.group(1)

					input_name = join(dir, "proj_%s.png" % angle)
					target_name = join(dir, "view_%s.png" % angle)

					with tf.gfile.FastGFile(input_name, 'rb') as input_file, \
						tf.gfile.FastGFile(target_name, 'rb') as target_file:

						input_image = input_file.read()
						target_image = target_file.read()

						example = tf.train.Example(features=tf.train.Features(feature={
							'collection': _bytes_feature(tf.compat.as_bytes(synset_id)),
							'model': _bytes_feature(tf.compat.as_bytes(id)),
							'angle': _float_feature(float(angle)),
							'input_image': _bytes_feature(tf.compat.as_bytes(input_image)),
							'target_image': _bytes_feature(tf.compat.as_bytes(target_image))}))
    
					records.write(example.SerializeToString())
					print("Found %s" % join(synset_id, id))
				except:
					pass

if __name__ == '__main__':
	main(sys.argv[1:])
