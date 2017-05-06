#!/bin/bash

if [ $# -lt 1 ]
then
	echo 1>&2 "Usage: run.sh <collection id>"
	exit 1
fi

basepath="C:/Users/Chris/OneDrive/Documents/ACS/Project/NewViewDNN"
find "$basepath/data/shapenet/$1" -maxdepth 1 -type d | parallel --retries 3 --joblog preprocess.log --max-lines 20 blender -b -P "$basepath/utils/view_interpolate.py" --
