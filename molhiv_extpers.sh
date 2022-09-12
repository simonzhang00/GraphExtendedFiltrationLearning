#!/bin/bash

for iter in {0..0}
  do
  dataset="molhiv"
  name=${dataset}-${iter}
  python3 -m train.mol_train_eval --lr 0.001 --dataset_name ogbg-molhiv --readout extph --filt_conv_number 2 --num_epochs 100 --batch_size 128 --use_raw_node_label True --output_dir ./results --device 0 --exp_name ${name} #&> dd-sum-len-2-slurm-${SLURM_JOB_ID}.out
  done
