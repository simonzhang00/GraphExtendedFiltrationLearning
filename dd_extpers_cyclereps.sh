#!/bin/bash

for iter in {0..0}
  do
  dataset="dd"
  name=${dataset}-${iter}
  python3 -m train.dd_train_eval --filt_conv_number 2 --readout extph_cyclereps --num_epochs 100 --batch_size 128 --use_raw_node_label True --output_dir ./results --device 0 --exp_name ${name} #&> dd-len-2-slurm-${SLURM_JOB_ID}.out
  done
