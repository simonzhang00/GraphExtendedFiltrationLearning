#!/bin/bash

for len in {1..5}
  do
  dataset="imdb-multi"
  name=${dataset}-${len}
  python3 -m train.imdb-multi_train_eval --filt_conv_number ${len} --num_epochs 200 --batch_size 128 --output_dir ./results --device 0 --exp_name ${name} &> imdb-multi-len-${len}.out
  done