export CUDA_VISIBLE_DEVICES=0

cd /root/autodl-tmp/p01_code/projection_predict

for loss_method in MSE
do
python main.py \
    --exp_id EXP_2024.12.1.9 \
    --exp_group stage_8-16\
    --exp_name $loss_method\
    --net unet\
    --data_path /root/autodl-tmp/shared_data/projection/raw\
    --device cuda\
    --epochs 600\
    --batch_size 64\
    --lr 8e-4\
    --dropout 0\
    --scheduler True\
    --early_stop True \
    --check_point 0\
    --stage 8-16\
    --loss_method $loss_method\
    --wandb disabled\

echo '====================================================================================================================='
done