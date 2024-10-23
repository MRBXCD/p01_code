import matplotlib.pyplot as plt
import skimage.metrics as sk
import os
import numpy as np
import glob

def data_load():
    raw = []
    dl = []

    for index in range(4):
        sequence_length = 2 ** (index+4)
        data = np.load(f'/home/mrb2/experiments/graduation_project/shared_data/voxel/recons/raw/Val_Recons_SIRT_{sequence_length}_angles.npz')['arr_0']
        raw.append(data)
    raw = np.array(raw)

    for index in range(3):
        sequence_length = 2 ** (index+3)
        data = np.load(f'/home/mrb2/experiments/graduation_project/shared_data/voxel/recons/dl/Val_Recons_SIRT_{sequence_length}-{sequence_length*2}_angles.npz')['arr_0']
        dl.append(data)
    dl = np.array(dl)

    # dl = np.load("/home/mrb2/experiments/graduation_project/shared_data/voxel/recons/dl/Val_Recons_SIRT_8-16_angles.npz")['arr_0']
    # dl = np.expand_dims(dl, axis=0)
    print(f'shape of raw recons voxels {raw.shape}, shape of dl improved recons {dl.shape}')
    return raw, dl


def plot_data(data1, data2, x_labels=None):
    plt.figure(figsize=(6, 5))
    if x_labels:
        plt.plot(x_labels, data1, marker='o')
        plt.plot(x_labels, data2, marker='o')
        plt.xticks()
    else:
        plt.plot(data1, marker='o')
    plt.xlabel('Reconstruction Views')
    plt.ylabel('RMSE')
    # plt.grid(True)
    plt.savefig('./raw.png')

def rmse_calc_raw(data): 
    voxel2 = np.load('/home/mrb2/experiments/graduation_project/shared_data/voxel/raw/Val_LIDC_128.npz')['arr_0']
    rmses = np.zeros(5)
    rmses[0] = 5.205176743831869263e-02 # RMSE of 8 view recons voxel and raw voxel

    for index in range(4):
        voxel1 = data[index].squeeze()
        rmse = sk.normalized_root_mse(voxel1, voxel2, normalization='min-max')
        rmses[index+1] = rmse
        
    rmses.tolist()
    return rmses

def rmse_calc_dl(data): 
    voxel2 = np.load('/home/mrb2/experiments/graduation_project/shared_data/voxel/raw/Val_LIDC_128.npz')['arr_0']
    rmses = np.zeros(5)
    rmses[0] = 5.205176743831869263e-02 # RMSE of 8 view recons voxel and raw voxel

    for index in range(3):
        voxel1 = data[index].squeeze()
        rmse = sk.normalized_root_mse(voxel1, voxel2, normalization='min-max')
        rmses[index+1] = rmse
   
    zero_indices = [i for i, x in enumerate(rmses) if x == 0]
    rmses[zero_indices] = rmse
        
    rmses.tolist()
    return rmses

raw, dl = data_load()
data_raw = rmse_calc_raw(raw)
data_proj = rmse_calc_dl(dl)
np.savetxt('rmse_raw.txt', data_raw)
np.savetxt('rmse_proj.txt', data_proj)
# 自定义横坐标
x_labels = ['8', '16', '32', '64', '128']  # 替换为你的自定义横坐标

plot_data(data_raw, data_proj, x_labels)
