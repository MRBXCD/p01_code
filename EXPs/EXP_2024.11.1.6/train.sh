export CUDA_VISIBLE_DEVICES=0

cd /home/mrb2/experiments/graduation_project/p01_code/projection_predict

for loss_method in Combined_loss
do

python main.py \
    --net atten_unet\
    --exp_id EXP_2024.11.1.5 \
    --data_path /home/mrb2/experiments/graduation_project/shared_data/projection/raw\
    --device cuda\
    --epochs 200\
    --batch_size 8\
    --lr 1e-4\
    --scheduler True\
    --early_stop True \
    --check_point 0\
    --stage 8-16\
    --loss_method $loss_method\
    --dropout 0.1\
    --wandb disabled

echo '====================================================================================================================='
done