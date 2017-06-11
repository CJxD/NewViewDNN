#1/bin/bash

data_path="../data"

script="../src/models/run-autoencoder.py"
flags="$* -d"
train_flags="train -i $data_path/train-nc.tfrecords --summary-interval 500 $flags"
val_flags="validate -i $data_path/val-nc.tfrecords --summary-interval 0 $flags"
run_flags="run -i $data_path/test_single.txt --summary-interval 0 $flags"

echo "========== Training ========="
#$script $train_flags
echo "========= Validating ========"
#$script $val_flags
echo "========== Testing =========="
$script $run_flags

