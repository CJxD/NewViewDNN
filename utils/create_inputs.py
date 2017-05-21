#!/usr/bin/env python3

import sys
import re
from os.path import *
from glob import glob
base_path = dirname(dirname(realpath(__file__)))

usage = "Usage: create_inputs.py [collection id, ...]"

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
	
	with \
		open(join(base_path, "data", ".shapenet", "splits.csv"), 'r') as splits, \
		open(join(base_path, "data", "train_proj.txt"), 'w') as train_proj, \
		open(join(base_path, "data", "train_params.txt"), 'w') as train_params, \
		open(join(base_path, "data", "train_out.txt"), 'w') as train_out, \
		open(join(base_path, "data", "val_proj.txt"), 'w') as val_proj, \
		open(join(base_path, "data", "val_params.txt"), 'w') as val_params, \
		open(join(base_path, "data", "val_out.txt"), 'w') as val_out, \
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
			
			if collections is not None and synsetId not in collections:
				continue
				
			id = entry[3]
			split = entry[4]
			dir = join(base_path, "data", ".shapenet", '0' + str(synsetId), id, "renders")

			if split == "train":
				proj = train_proj
				params = train_params
				out = train_out
			elif split == "val":
				proj = val_proj
				params = val_params
				out = val_out
			elif split == "test":
				proj = test_proj
				params = test_params
				out = test_out
			else:
				continue

			for file in glob(join(dir, "proj_*.png")):
				try:
					matches = proj_pat.match(file)
					angle = matches.group(1)

					print(str(join(dir, "proj_%s.png" % angle)), file=proj)
					print(str(join(dir, "view_%s.png" % angle)), file=out)
					print(angle, file=params)
				except:
					pass

if __name__ == '__main__':
	main(sys.argv[1:])
