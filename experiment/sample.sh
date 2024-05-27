#!/bin/bash

py=~/your/path/here/python

$py -u train_and_eval.py \
            --batch_size 16 \
            --lr 1e-4 \
            --epoch_num 100 \
            --num_embeddings 6 \
            --d_model 20 \
            --h 4 \
            --d_k 5 \
            --d_v 5 \
            --d_ff 2048 \
            --N 3 \
            --seq_len 500 \
            --padding_idx 5 \
            --dataset_root_path '/your/path/here/RNA8F' \
            --sub_dataset_list 'short:medium' \
            --result_path 'results' \
            --model_and_log_main_name 'sample' \
            --train_use_class_indices_target
