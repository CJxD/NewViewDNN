#!/usr/bin/env python3

import sys
from os.path import *
base_path = dirname(dirname(realpath(__file__)))

def main(args):
	if len(args) > 1:
		print >> sys.stderr, "Usage: create_inputs.py [collection id]"
		return 1
		
	if len(args) == 1:
		collection = int(args[0])
	else:
		collection = None
	
	with \
		open(join(base_path, "data", "shapenet", "splits.csv"), 'r') as splits, \
		open(join(base_path, "data", "train_camA.txt"), 'w') as train_camA, \
		open(join(base_path, "data", "train_camB.txt"), 'w') as train_camB, \
		open(join(base_path, "data", "train_camO.txt"), 'w') as train_camO, \
		open(join(base_path, "data", "train_params.txt"), 'w') as train_params, \
		open(join(base_path, "data", "test_camA.txt"), 'w') as test_camA, \
		open(join(base_path, "data", "test_camB.txt"), 'w') as test_camB, \
		open(join(base_path, "data", "test_camO.txt"), 'w') as test_camO, \
		open(join(base_path, "data", "test_params.txt"), 'w') as test_params:
		header = None
		for line in splits:
			if not header:
				header = line
				continue
				
			entry = line.strip().split(',')
			synsetId = int(entry[1])
			
			if collection and synsetId != collection:
				continue
				
			id = entry[3]
			split = entry[4]
			dir = join(base_path, "data", "shapenet", '0' + str(synsetId), id, "models")
			
			if not exists(join(dir, "model_normalized_0.5.png")):
				continue

			if split == "train":
				camA = train_camA
				camB = train_camB
				camO = train_camO
				params = train_params
			elif split == "val":
				camA = test_camA
				camB = test_camB
				camO = test_camO
				params = test_params
			elif split == "test":
				continue
			else:
				continue
				
			print(str(join(dir, "model_normalized_CamA.png")), file=camA)
			print(str(join(dir, "model_normalized_CamB.png")), file=camB)
			print(str(join(dir, "model_normalized_0.5.png")), file=camO)
			print(str(0.5), file=params)

if __name__ == '__main__':
	main(sys.argv[1:])