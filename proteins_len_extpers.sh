#!/bin/bash

for len in {2..5}
  do
  dataset="proteins"
  name=${dataset}-${len}
  python3 -m train.proteins_train_eval --filt_conv_number ${len} --num_epochs 50 --batch_size 128 --use_raw_node_label True --output_dir ./results --device 0 --exp_name ${name} &> proteins-len-${len}.out
  done
