#!/usr/bin/env python3

import sys, os
from subprocess import call
from glob import glob

models_per_run = 100
parallel_processes = 1

script = "C:/Users/Chris/OneDrive/Documents/ACS/Project/NewViewDNN/utils/view_interpolate.py"
data = "Z:/ShapeNetCore.v2"
cmd = ['blender-2.69/blender', '-b', '-P', script, '--']

def main(args):
	models = glob(os.path.join(data, args[0], '*'))
	
	while len(models) > 0:
		call(cmd + models[:models_per_run])
		
		with open('view_interpolate.log', 'r') as log:
			for model in log:
				try:
					models.remove(model)
				except:
					pass

if __name__ == "__main__":
	argv = sys.argv[1:]
	main(argv)