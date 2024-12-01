#!/bin/bash

B_train=/root/autodl-tmp/shared_data/projection/backproj/Projection_train_data_16_angles_padded.npz
B_val=/root/autodl-tmp/shared_data/projection/backproj/Projection_val_data_16_angles_padded.npz
B2_train=/root/autodl-tmp/p01_code/EXPs/EXP_2024.12.1.7/train.npz
B2_val=/root/autodl-tmp/p01_code/EXPs/EXP_2024.12.1.7/val.npz
D_train=/root/autodl-tmp/shared_data/projection/dl/projections_train_8-16.npz
D_val=/root/autodl-tmp/shared_data/projection/dl/projections_val_8-16.npz

for train_proj in "$B2_train" "$D_train"
do
    if [ "$train_proj" = "$B_train" ]; then
        val_proj=$B_val
    elif [ "$train_proj" = "$B2_train" ]; then
        val_proj=$B2_val
    elif [ "$train_proj" = "$D_train" ]; then
        val_proj=$D_val
    else
        echo "未识别的训练路径: $train_proj"
        continue
    fi

    python3 /root/autodl-tmp/p01_code/reconstruction/main.py \
        --operation reconstruction \
        --n_angle 16 \
        --projection_path_train $train_proj \
        --projection_path_val $val_proj \

echo '------------------------------------------------------------------------------------------------------------'
done