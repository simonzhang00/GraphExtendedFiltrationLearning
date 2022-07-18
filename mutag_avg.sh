#!/bin/bash

for iter in {0..0}
  do
  dataset="mutag"
  name=${dataset}-${iter}
  python3 -m train.mutag_train_eval --readout average --filt_conv_number 1 --num_epochs 100 --batch_size 128 --output_dir ./results --device 0 --exp_name ${name} #&> mutag-len-2-slurm-${SLURM_JOB_ID}.out
  done
