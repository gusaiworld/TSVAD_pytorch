#!/bin/bash

python main.py \
--train_list /home/users/ntu/adnan002/scratch/DIHARD3/third_dihard_challenge_dev/data/ts_dev.json \
--eval_list /home/users/ntu/adnan002/scratch/DIHARD3/third_dihard_challenge_eval/data/ts_eval.json \
--train_path /home/users/ntu/adnan002/scratch/DIHARD3/third_dihard_challenge_dev/data \
--eval_path /home/users/ntu/adnan002/scratch/DIHARD3/third_dihard_challenge_eval/data \
--save_path exps/res23 \
--rs_len 4 \
--test_shift 0.5 \
--min_silence 0.32 \
--min_speech 0.00 \
--threshold 0.50 \
--n_cpu 12 \
--eval \
--init_model /home/users/ntu/adnan002/scratch/TSVAD_pytorch/ts-vad/pretrained_models/ts-vad.model \