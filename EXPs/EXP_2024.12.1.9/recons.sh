#!/bin/bash

# train_proj=/home/mrb2/experiments/graduation_project/p01_code/projection_predict/result/extraction/4-8/EXP_2024.12.1.10_L1_model_output_train.npz
# val_proj=/home/mrb2/experiments/graduation_project/p01_code/projection_predict/result/extraction/4-8/EXP_2024.12.1.10_L1_model_output_val.npz

train_proj=/root/autodl-tmp/p01_code/projection_predict/result/extraction/8-16/EXP_2024.12.1.9_L1_model_output_train.npz
val_proj=/root/autodl-tmp/p01_code/projection_predict/result/extraction/8-16/EXP_2024.12.1.9_L1_model_output_val.npz

cd /root/autodl-tmp/p01_code/reconstruction

python3 main.py \
    --operation reconstruction \
    --n_angle 16 \
    --projection_path_train $train_proj \
    --projection_path_val $val_proj \

echo '------------------------------------------------------------------------------------------------------------'
