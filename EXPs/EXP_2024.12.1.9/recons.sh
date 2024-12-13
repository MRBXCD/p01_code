#!/bin/bash

# train_proj=/home/mrb2/experiments/graduation_project/p01_code/projection_predict/result/extraction/4-8/EXP_2024.12.1.10_L1_model_output_train.npz
# val_proj=/home/mrb2/experiments/graduation_project/p01_code/projection_predict/result/extraction/4-8/EXP_2024.12.1.10_L1_model_output_val.npz

train_proj=/home/mrb2/experiments/graduation_project/p01_code/reconstruction/result/projection/4_angles/Projection_train_data_4_angles_padded.npz
val_proj=/home/mrb2/experiments/graduation_project/p01_code/reconstruction/result/projection/4_angles/Projection_val_data_4_angles_padded.npz

cd /home/mrb2/experiments/graduation_project/p01_code/reconstruction

python3 main.py \
    --operation reconstruction \
    --n_angle 4 \
    --projection_path_train $train_proj \
    --projection_path_val $val_proj \

echo '------------------------------------------------------------------------------------------------------------'
