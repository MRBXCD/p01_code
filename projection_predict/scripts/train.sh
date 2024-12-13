export CUDA_VISIBLE_DEVICES=0

cd /home/mrb2/experiments/graduation_project/p01_code/projection_predict

for loss_method in Combined_loss
do
python main.py \
    --exp_id EXP_2024.12.1.9 \
    --exp_group arch_v2.5\
    --exp_name $loss_method\
    --net unet\
    --data_path /home/mrb2/experiments/graduation_project/shared_data/projection/raw\
    --device cuda\
    --epochs 200\
    --batch_size 8\
    --lr 1e-4\
    --dropout 0\
    --scheduler True\
    --early_stop True \
    --check_point 0\
    --stage 8-16\
    --loss_method $loss_method\
    --wandb online\

echo '====================================================================================================================='
done