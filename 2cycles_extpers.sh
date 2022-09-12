#!/bin/bash

for iter in {0..0}
  do
  dataset="2cycles"
  name=${dataset}-${iter}
  python3 -m train.2cycles_train_eval --readout extph --num_epochs 100 --batch_size 128 --output_dir ./results --device 0 --exp_name ${name} #&> imdb-multi-slurmfiles/slurm-${SLURM_JOB_ID}.out
  done