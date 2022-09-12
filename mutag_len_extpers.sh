#!/bin/bash

for len in {1..5}
  do
  dataset="mutag"
  name=${dataset}-${len}
  python3 -m train.mutag_train_eval --filt_conv_number ${len} --num_epochs 100 --batch_size 128 --use_raw_node_label True --output_dir ./results --device 0 --exp_name ${name} &> mutag-len-${len}.out
  done