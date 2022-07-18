#!/bin/bash
for iter in {0..0}
  do
  dataset="pinwheels"
  name=${dataset}-${iter}
  python3 -m train.pinwheels_train_eval --readout max --num_epochs 100 --batch_size 128 --output_dir ./results --device 0 --exp_name ${name} #&> imdb-multi-slurmfiles/slurm-${SLURM_JOB_ID}.out
  done
