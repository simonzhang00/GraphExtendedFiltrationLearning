#!/bin/bash

for iter in {0..0}
  do
  dataset="molbace"
  name=${dataset}-${iter}
  python3 -m train.mol_train_eval --dataset_name ogbg-molbace --lr 0.001 --readout extph --filt_conv_number 2 --num_epochs 100 --batch_size 128 --use_raw_node_label True --output_dir ./results --device 0 --exp_name ${name} #&> dd-sum-len-2-slurm-${SLURM_JOB_ID}.out
  done
