#!/bin/bash

script="./run-test.sh"

$script -x 32 -y 32 -C 3 --model-file checkpoints/32x32x3/model.ckpt -o outputs/32x32x3 | tee 32x32x3.log

$script -x 32 -y 32 -C 3 --filter-sizes 7 7 7 --model-file checkpoints/32x32x3-777/model.ckpt -o outputs/32x32x3-777 | tee 32x32x3-777.log

$script -x 32 -y 32 -C 3 --filter-sizes 3 3 3 3 3 --num-filters 10 10 10 10 10 --model-file checkpoints/32x32x3-5deep/model.ckpt -o outputs/32x32x3-5deep | tee 32x32x3-5deep.log

$script -x 32 -y 32 -C 3 -d --model-file checkpoints/32x32x3-diff/model.ckpt -o outputs/32x32x3-diff | tee 32x32x3-diff.log

$script -x 32 -y 32 -C 4 --model-file checkpoints/32x32x4/model.ckpt -o outputs/32x32x4 | tee 32x32x4.log

$script -x 64 -y 64 -C 3 --model-file checkpoints/64x64x3/model.ckpt -o outputs/64x64x3 | tee 64x64x3.log

$script -x 128 -y 128 -C 3 --model-file checkpoints/128x128x3/model.ckpt -o outputs/128x128x3 | tee 128x128x3.log

$script -x 256 -y 256 -C 3 --model-file checkpoints/256x256x3/model.ckpt -o outputs/256x256x3 | tee 256x256x3.log

$script -X 512 -Y 512 -x 512 -y 512 -C 3 --model-file checkpoints/512x512x3/model.ckpt -o outputs/512x512x3 | tee 512x512x3.log
