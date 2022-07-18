#!/bin/bash

for iter in {0..0}
  do
  dataset="proteins"
  name=${dataset}${iter}
  python3 -m train.proteins_train_eval --num_epochs 100 --batch_size 128 --use_raw_node_label True --output_dir ./results --device 0 --exp_name ${name} #&> imdb-multi-slurmfiles/slurm-${SLURM_JOB_ID}.out
  done
