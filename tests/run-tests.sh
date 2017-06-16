#!/bin/bash

script="./run-test.sh"
data_path="../data"

log="tee -a"

if [ $# -eq 1 ]; then
	base="$1"
else
	base="."
fi

$script -x 32 -y 32 -C 3 --model-file $base/checkpoints/32x32x3/model.ckpt --log-dir $base/logs/32x32x3 -o $base/outputs/32x32x3 | $log $base/32x32x3.log

$script -x 32 -y 32 -C 3 --filter-sizes 7 7 7 --model-file $base/checkpoints/32x32x3-777/model.ckpt --log-dir $base/logs/32x32x3-777 -o $base/outputs/32x32x3-777 | $log $base/32x32x3-777.log

$script -x 32 -y 32 -C 3 --filter-sizes 3 3 3 3 3 --num-filters 10 10 10 10 10 --model-file $base/checkpoints/32x32x3-5deep/model.ckpt --log-dir $base/logs/32x32x3-5deep -o $base/outputs/32x32x3-5deep | $log $base/32x32x3-5deep.log

$script -x 32 -y 32 -C 3 -d --model-file $base/checkpoints/32x32x3-diff/model.ckpt --log-dir $base/logs/32x32x3-diff -o $base/outputs/32x32x3-diff | $log $base/32x32x3-diff.log

$script -x 32 -y 32 -C 4 --model-file $base/checkpoints/32x32x4/model.ckpt --log-dir $base/logs/32x32x4 -o $base/outputs/32x32x4 | $log $base/32x32x4.log

$script -x 64 -y 64 -C 3 --model-file $base/checkpoints/64x64x3/model.ckpt --log-dir $base/logs/64x64x3 -o $base/outputs/64x64x3 | $log $base/64x64x3.log

$script -x 128 -y 128 -C 3 --model-file $base/checkpoints/128x128x3/model.ckpt --log-dir $base/logs/128x128x3 -o $base/outputs/128x128x3 | $log $base/128x128x3.log

$script -x 256 -y 256 -C 3 --model-file $base/checkpoints/256x256x3/model.ckpt --log-dir $base/logs/256x256x3 -o $base/outputs/256x256x3 | $log $base/256x256x3.log

#$script -X 512 -Y 512 -x 512 -y 512 -C 3 --model-file $base/checkpoints/512x512x3/model.ckpt --log-dir $base/logs/512x512xx3 -o $base/outputs/512x512x3 | $log $base/512x512x3.log
