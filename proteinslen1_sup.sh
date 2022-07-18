#!/bin/bash

for len in {1..1}
  do
  dataset="proteins"
  name=${dataset}-${len}
  python3 -m train.proteins_train_eval --filt_conv_number ${len} --num_epochs 50 --batch_size 128 --use_raw_node_label True --output_dir ./results --device 0 --exp_name ${name} #&> len-${len}-slurm-${SLURM_JOB_ID}.out
  done
