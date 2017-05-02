#!/bin/bash

if [ $# -lt 1 ]
then
	echo 1>&2 "Usage: run.sh <path to collection>"
	exit 1
fi

for model in $1/*
do
	blender -b -P C:/Users/Chris/OneDrive/Documents/ACS/Project/NewViewDNN/utils/view_interpolate.py -- $model
done
