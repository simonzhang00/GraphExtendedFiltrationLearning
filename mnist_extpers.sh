#!/bin/bash

for iter in {0..0}
  do
  dataset="dd"
  name=${dataset}-${iter}
  python3 -m train.mnist_train_eval --filt_conv_number 2 --num_epochs 1000 --batch_size 128 --lr 5e-4 --use_raw_node_label True --output_dir ./results --device 0 --exp_name ${name} #&> dd-len-2-slurm-${SLURM_JOB_ID}.out
  done
