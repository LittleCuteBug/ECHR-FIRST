#!/bin/bash

path_root="/home/huyqnguyen/Documents/echr-first"
eval_batch_size=8

for dataset in "ecthr_a" "ecthr_b"; do
  for learning_rate in 1e-05; do
    for train_batch_size in 1; do
      for seed_number in 43 97; do
        for task in "base" "add_true_label" "add_false_label"; do
          command="python longformer_base.py --task_name ${task} --dataset ${dataset} \
          --learning_rate ${learning_rate} \
          -p ${path_root} \
          --seed_number ${seed_number} \
          --train_batch_size ${train_batch_size} \
          --eval_batch_size ${eval_batch_size} \
          --test"
          ${command}
        done
      done
    done
  done
done
