import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import root_mean_squared_error as mse
import pandas as pd

# 定义补全路径函数，仅当路径为相对路径时进行补全
def complete_path(file_path):
    base_dir = '/home/mrb2/experiments/graduation_project/shared_data'
    return os.path.join(base_dir, file_path.lstrip('/'))

# 定义体素比较函数
def compare_voxels(raw_voxel_path, rr_voxel_path, dl_voxel_path, output_dir, index):
    # 加载体素数据
    print(f'Loading raw voxels from path {raw_voxel_path}')
    raw_voxel = np.load(raw_voxel_path)['arr_0']
    print(f'Loading raw recons voxels from path {rr_voxel_path}')
    rr_voxel = np.load(complete_path(rr_voxel_path))['arr_0']
    print(f'Loading dl voxels from path {dl_voxel_path} \n -----------------------------------------------------------------------------')
    dl_voxel = np.load(complete_path(dl_voxel_path))['arr_0']
    raw_voxel_item = raw_voxel[0]
    rr_voxel_item = rr_voxel[0]
    dl_voxel_item = dl_voxel[0]
    
    if 'train' in raw_voxel_path.lower():
        prefix = 'train'
    else:
        prefix = 'val'

    # 获取切片
    slices = [(64, slice(None), slice(None)), (slice(None), 64, slice(None)), (slice(None), slice(None), 64)]
    raw_slices = [raw_voxel_item[s] for s in slices]
    rr_slices = [rr_voxel_item[s] for s in slices]
    dl_slices = [dl_voxel_item[s] for s in slices]
    
    # 显示并保存切片图像
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        axes[i, 0].imshow(raw_slices[i], cmap='gray')
        axes[i, 0].set_title(f'Raw Slice {i+1}')
        axes[i, 1].imshow(rr_slices[i], cmap='gray')
        axes[i, 1].set_title(f'RR Slice {i+1}')
        axes[i, 2].imshow(dl_slices[i], cmap='gray')
        axes[i, 2].set_title(f'DL Slice {i+1}')
    plt.tight_layout()
    voxel_img_path = os.path.join(output_dir, f'{prefix}_voxel_{index}.png')
    plt.savefig(voxel_img_path, dpi=500)
    plt.close()

    # 计算并返回指标 (RMSE, SSIM, PSNR)
    metrics = {'Index': index}
    comparisons = [('raw', 'rr', raw_voxel_item, rr_voxel_item), 
                   ('raw', 'dl', raw_voxel_item, dl_voxel_item), 
                   ('rr', 'dl', rr_voxel_item, dl_voxel_item)]
    
    rmse_values = []
    ssim_values = []
    psnr_values = []
    
    for name1, name2, v1, v2 in comparisons:
        rmse_value = np.sqrt(mse(v1.flatten(), v2.flatten()))
        ssim_value = ssim(v1, v2, data_range=v2.max() - v2.min())
        psnr_value = psnr(v1, v2, data_range=v2.max() - v2.min())
        rmse_values.append(f'{rmse_value:.4f}')
        ssim_values.append(f'{ssim_value:.4f}')
        psnr_values.append(f'{psnr_value:.4f}')

        # 恢复保存到 txt 文件的功能
        metrics_txt_path = os.path.join(output_dir, f'{prefix}_voxel_metrics_{index}.txt')
        with open(metrics_txt_path, 'a') as f:
            f.write(f'{name1}_vs_{name2}:\n')
            f.write(f'RMSE: {rmse_value:.4f}\n')
            f.write(f'SSIM: {ssim_value:.4f}\n')
            f.write(f'PSNR: {psnr_value:.4f}\n')
            f.write('\n')
    
    # 将同一指标的多个比较结果合并到一个字符串中，用斜杠分隔
    metrics['RMSE'] = ' / '.join(rmse_values)
    metrics['SSIM'] = ' / '.join(ssim_values)
    metrics['PSNR'] = ' / '.join(psnr_values)
    metrics['Prefix'] = prefix  # 添加前缀用于区分 train 和 val
    
    metrics_df = pd.DataFrame([metrics])
    return metrics_df  # 返回指标数据框

def compare_projection(raw_projection_path, dl_projection_path, output_dir, index):
    # 加载投影数据
    raw_projection = np.load(complete_path(raw_projection_path))['arr_0']
    dl_projection = np.load(complete_path(dl_projection_path))['arr_0']

    length = raw_projection.shape[2]
    raw_projection_item = raw_projection[0].reshape(148, length*148)
    dl_projection_item = dl_projection[0].reshape(148, length*148)
    
    # 显示并保存投影图像
    if 'train' in raw_projection_path.lower():
        prefix = 'train'
    else:
        prefix = 'val'

    fig, axes = plt.subplots(2, 1, figsize=(12, 3))

    axes[0].imshow(raw_projection_item, cmap='gray')
    axes[0].set_title(f'Raw Projection')
    axes[1].imshow(dl_projection_item, cmap='gray')
    axes[1].set_title(f'DL Projection')
    
    plt.tight_layout()
    voxel_img_path = os.path.join(output_dir, f'{prefix}_projection_{index}.png')
    plt.savefig(voxel_img_path, dpi=500)
    plt.close()

    # 计算并返回指标 (RMSE, SSIM, PSNR)
    metrics = {'Index': index, 'Prefix': prefix}
    comparisons = [('raw', 'dl', raw_projection[0], dl_projection[0])]
    
    rmse_values = []
    ssim_values = []
    psnr_values = []
    
    for name1, name2, v1, v2 in comparisons:
        rmse_value = np.sqrt(mse(v1.flatten(), v2.flatten()))
        ssim_value = ssim(v1, v2, data_range=v2.max() - v2.min())
        psnr_value = psnr(v1, v2, data_range=v2.max() - v2.min())
        rmse_values.append(f'{rmse_value:.4f}')
        ssim_values.append(f'{ssim_value:.4f}')
        psnr_values.append(f'{psnr_value:.4f}')

        # 恢复保存到 txt 文件的功能
        metrics_txt_path = os.path.join(output_dir, f'{prefix}_projection_metrics_{index}.txt')
        with open(metrics_txt_path, 'a') as f:
            f.write(f'{name1}_vs_{name2}:\n')
            f.write(f'RMSE: {rmse_value:.4f}\n')
            f.write(f'SSIM: {ssim_value:.4f}\n')
            f.write(f'PSNR: {psnr_value:.4f}\n')
            f.write('\n')
    
    # 将同一指标的多个比较结果合并到一个字符串中，用斜杠分隔
    metrics['RMSE'] = ' / '.join(rmse_values)
    metrics['SSIM'] = ' / '.join(ssim_values)
    metrics['PSNR'] = ' / '.join(psnr_values)
    
    metrics_df = pd.DataFrame([metrics])
    return metrics_df  # 返回指标数据框

# 从文件对加载并处理
def process_voxel_file_pairs(pair_file_path, output_dir):
    # 定义原始体素文件路径
    raw_voxel_files = {
        'train': '/home/mrb2/experiments/graduation_project/shared_data/voxel/raw/Train_LIDC_128.npz',
        'val': '/home/mrb2/experiments/graduation_project/shared_data/voxel/raw/Val_LIDC_128.npz'
    }
    
    all_metrics_train = []  # 用于存储训练集的指标数据
    all_metrics_val = []    # 用于存储验证集的指标数据
    with open(pair_file_path, 'r') as f:
        lines = f.readlines()
        
        index = 16
        for line in lines:
            file_pair = line.strip().split()
            if len(file_pair) == 2:
                dl_voxel_path, rr_voxel_path = file_pair
                
                # 根据文件路径判断是Train还是Val
                if 'Train' in rr_voxel_path or 'train' in rr_voxel_path:
                    dataset_type = 'train'
                    raw_voxel_path = raw_voxel_files['train']
                elif 'Val' in rr_voxel_path or 'val' in rr_voxel_path:
                    dataset_type = 'val'
                    raw_voxel_path = raw_voxel_files['val']
                else:
                    print(f'Unknown dataset type in: {rr_voxel_path}')
                    continue

                print(f'Processing: {dl_voxel_path} and {rr_voxel_path}')
                metrics_df = compare_voxels(raw_voxel_path, rr_voxel_path, dl_voxel_path, output_dir, index)
                
                # 根据数据集类型，分别保存指标数据
                if dataset_type == 'train':
                    all_metrics_train.append(metrics_df)
                else:
                    all_metrics_val.append(metrics_df)
            else:
                print(f'Skipping incomplete pair: {line.strip()}')
            index *= 2

    # 合并并保存训练集的指标数据
    if all_metrics_train:
        combined_metrics_df_train = pd.concat(all_metrics_train, ignore_index=True)
        combined_metrics_df_train.to_csv(os.path.join(output_dir, 'voxel_metrics_train.csv'), index=False)

    # 合并并保存验证集的指标数据
    if all_metrics_val:
        combined_metrics_df_val = pd.concat(all_metrics_val, ignore_index=True)
        combined_metrics_df_val.to_csv(os.path.join(output_dir, 'voxel_metrics_val.csv'), index=False)

def process_projection_file_pairs(pair_file_path, output_dir):

    all_metrics = []  # 用于存储所有指标数据
    with open(pair_file_path, 'r') as f:
        lines = f.readlines()
        
        index = 16
        for line in lines:
            file_pair = line.strip().split()
            if len(file_pair) == 2:
                dl_projection_path, raw_projection_path = file_pair
                print(f'Processing: {raw_projection_path} and {dl_projection_path}')
                metrics_df = compare_projection(raw_projection_path, dl_projection_path, output_dir, index)
                all_metrics.append(metrics_df)
            else:
                print(f'Skipping incomplete pair: {line.strip()}')
            index *= 2
            if index == 128:
                index = 16

    # 合并所有指标数据并保存为一个CSV文件
    if all_metrics:
        combined_metrics_df = pd.concat(all_metrics, ignore_index=True)
        combined_metrics_df.to_csv(os.path.join(output_dir, 'projection_metrics_overall.csv'), index=False)

# 示例调用
pair_file_path1 = './voxel_file_pair.txt'
output_dir = './result/overall_comparation/'

pair_file_path2 = './projection_file_pair.txt'
output_dir = './result/overall_comparation/'

os.makedirs(output_dir, exist_ok=True)
process_voxel_file_pairs(pair_file_path1, output_dir)
process_projection_file_pairs(pair_file_path2, output_dir)
