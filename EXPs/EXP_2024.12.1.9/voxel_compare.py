import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse

voxel_4_dlR = np.load('/home/mrb2/experiments/graduation_project/p01_code/reconstruction/result/voxel/EXP_2024.12.1.9_train_SIRT_8_angles.npz')['arr_0']
#voxel_4_rawR = np.load('/home/mrb2/experiments/graduation_project/shared_data/voxel/recons/raw/Train_Recons_SIRT_4_angles.npz')['arr_0']
voxel_Raw = np.load('/home/mrb2/experiments/graduation_project/shared_data/voxel/raw/Train_LIDC_128.npz')['arr_0']


voxel_4_dlR = voxel_4_dlR.flatten()
voxel_Raw = voxel_Raw.flatten()
#rr_Raw = rmse(voxel_4_rawR, voxel_Raw)
dlr_Raw = rmse(voxel_4_dlR, voxel_Raw)

print(dlr_Raw)