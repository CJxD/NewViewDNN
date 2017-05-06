#!/usr/bin/env python3

import sys
from os.path import *
base_path = dirname(dirname(realpath(__file__)))

def main(args):
	if len(args) > 1:
		print("Usage: create_inputs.py [collection id]", file=sys.stderr)
		return 1
		
	if len(args) == 1:
		collection = int(args[0])
	else:
		collection = None
	
	with \
		open(join(base_path, "data", "shapenet", "splits.csv"), 'r') as splits, \
		open(join(base_path, "data", "train_cam0.txt"), 'w') as train_cam0, \
		open(join(base_path, "data", "train_cam1.txt"), 'w') as train_cam1, \
		open(join(base_path, "data", "train_proj.txt"), 'w') as train_proj, \
		open(join(base_path, "data", "train_params.txt"), 'w') as train_params, \
		open(join(base_path, "data", "train_out.txt"), 'w') as train_out, \
		open(join(base_path, "data", "val_cam0.txt"), 'w') as val_cam0, \
		open(join(base_path, "data", "val_cam1.txt"), 'w') as val_cam1, \
		open(join(base_path, "data", "val_proj.txt"), 'w') as val_proj, \
		open(join(base_path, "data", "val_params.txt"), 'w') as val_params, \
		open(join(base_path, "data", "val_out.txt"), 'w') as val_out, \
		open(join(base_path, "data", "test_cam0.txt"), 'w') as test_cam0, \
		open(join(base_path, "data", "test_cam1.txt"), 'w') as test_cam1, \
		open(join(base_path, "data", "test_proj.txt"), 'w') as test_proj, \
		open(join(base_path, "data", "test_params.txt"), 'w') as test_params, \
		open(join(base_path, "data", "test_out.txt"), 'w') as test_out:
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
			dir = join(base_path, "data", "shapenet", '0' + str(synsetId), id, "renders")
			
			if not exists(join(dir, "proj_1.0.png")):
				continue

			if split == "train":
				cam0 = train_cam0
				cam1 = train_cam1
				proj = train_proj
				params = train_params
				out = train_out
			elif split == "val":
				cam0 = val_cam0
				cam1 = val_cam1
				proj = val_proj
				params = val_params
				out = val_out
			elif split == "test":
				cam0 = test_cam0
				cam1 = test_cam1
				proj = test_proj
				params = test_params
				out = test_out
			else:
				continue
				
			print(str(join(dir, "view_0.0.png")), file=cam0)
			print(str(join(dir, "view_1.0.png")), file=cam1)
			print(str(join(dir, "proj_0.5.png")), file=proj)
			print(str(join(dir, "view_0.5.png")), file=out)
			print(str(0.5), file=params)

if __name__ == '__main__':
	main(sys.argv[1:])