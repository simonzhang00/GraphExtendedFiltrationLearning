#!/bin/bash

for iter in {0..0}
  do
  dataset="molhiv"
  name=${dataset}-${iter}
  python3 -m train.mol_train_eval --lr 0.001 --readout sum --dataset_name ogbg-molhiv --filt_conv_number 5 --filt_conv_dimension 300 --num_epochs 200 --batch_size 128 --use_node_deg False --use_raw_node_label True --output_dir ./results --device 0 --exp_name ${name} #&> dd-sum-len-2-slurm-${SLURM_JOB_ID}.out

  done
