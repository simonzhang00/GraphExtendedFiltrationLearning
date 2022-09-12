#!/bin/bash

for iter in {0..0}
  do
  dataset="imdb-multi"
  name=${dataset}-${iter}
  python3 -m train.imdb-multi_train_eval --readout sort --filt_conv_number 2 --num_epochs 200 --batch_size 128 --output_dir ./results --device 0 --exp_name ${name} #&> imdb-multi-len-2-slurm-${SLURM_JOB_ID}.out
  done
