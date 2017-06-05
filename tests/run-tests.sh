#!/bin/bash

data_path="../data"

script="../src/models/run-autoencoder.py"
flags=""
train_flags="train -i $data_path/train-nc.tfrecords --summary-interval 500 $flags"
val_flags="validate -i $data_path/val-nc.tfrecords --summary-interval 0 $flags"
run_flags="run -i $data_path/test_single.txt --summary-interval 0 $flags"

$script $train_flags -x 32 -y 32 -C 3 --model-file checkpoints/32x32x3/model.ckpt --log-dir logs/32x32x3
$script $val_flags -x 32 -y 32 -C 3 --model-file checkpoints/32x32x3/model.ckpt | tee 32x32x3-validation.log
$script $run_flags -x 32 -y 32 -C 3 --model-file checkpoints/32x32x3/model.ckpt -o outputs/32x32x3

$script $train_flags -x 32 -y 32 -C 3 -d --model-file checkpoints/32x32x3-diff/model.ckpt --log-dir logs/32x32x3-diff
$script $val_flags -x 32 -y 32 -C 3 -d --model-file checkpoints/32x32x3-diff/model.ckpt | tee 32x32x3-diff-validation.log
$script $run_flags -x 32 -y 32 -C 3 -d --model-file checkpoints/32x32x3-diff/model.ckpt -o outputs/32x32x3-d

$script $train_flags -x 32 -y 32 -C 4 --model-file checkpoints/32x32x4/model.ckpt --log-dir logs/32x32x4
$script $val_flags -x 32 -y 32 -C 4 --model-file checkpoints/32x32x4/model.ckpt | tee 32x32x4-validation.log
$script $run_flags -x 32 -y 32 -C 4 --model-file checkpoints/32x32x4/model.ckpt -o outputs/32x32x4

$script $train_flags -x 64 -y 64 -C 3 --model-file checkpoints/64x64x3/model.ckpt --log-dir logs/64x64x3
$script $val_flags -x 64 -y 64 -C 3 --model-file checkpoints/64x64x3/model.ckpt | tee 64x64x3-validation.log
$script $run_flags -x 64 -y 64 -C 3 --model-file checkpoints/64x64x3/model.ckpt -o outputs/64x64x3

$script $train_flags -x 128 -y 128 -C 3 --model-file checkpoints/128x128x3/model.ckpt --log-dir logs/128x128x3
$script $val_flags -x 128 -y 128 -C 3 --model-file checkpoints/128x128x3/model.ckpt | tee 128x128x3-validation.log
$script $run_flags -x 128 -y 128 -C 3 --model-file checkpoints/128x128x3/model.ckpt -o outputs/128x128x3

$script $train_flags -x 256 -y 256 -C 3 --model-file checkpoints/256x256x3/model.ckpt --log-dir logs/256x256x3
$script $val_flags -x 256 -y 256 -C 3 --model-file checkpoints/256x256x3/model.ckpt | tee 256x256x3-validation.log
$script $val_flags -x 256 -y 256 -C 3 --model-file checkpoints/256x256x3/model.ckpt -o outputs/256x256x3

$script $train_flags -X 512 -Y 512 -C 3 --model-file checkpoints/512x512x3/model.ckpt --log-dir logs/512x512x3
$script $val_flags -X 512 -Y 512 -C 3 --model-file checkpoints/512x512x3/model.ckpt | tee 512x512x3-validation.log
$script $run_flags -X 512 -Y 512 -C 3 --model-file checkpoints/512x512x3/model.ckpt -o outputs/512x512x3
