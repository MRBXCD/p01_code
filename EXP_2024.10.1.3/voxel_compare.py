import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse

voxel_A = np.load('/home/mrb2/experiments/graduation_project/shared_data/voxel/recons/raw/Train_Recons_SIRT_8_angles.npz')['arr_0']
voxel_C = np.load('/home/mrb2/experiments/graduation_project/shared_data/voxel/recons/backproj/Backproj_Recons_SIRT_16_angles.npz')['arr_0']
voxel_C2 = np.load('/home/mrb2/experiments/graduation_project/p01_code/reconstruction/result/voxel/voxel_c2.npz')['arr_0']
voxel_Raw = np.load('/home/mrb2/experiments/graduation_project/shared_data/voxel/raw/Train_LIDC_128.npz')['arr_0']

A_C = rmse(voxel_A.flatten(), voxel_C.flatten())
C2_Raw = rmse(voxel_C2.flatten(), voxel_Raw.flatten())
A_Raw = rmse(voxel_A.flatten(), voxel_Raw.flatten())
C_Raw = rmse(voxel_C.flatten(), voxel_Raw.flatten())

print(A_C, A_Raw, C_Raw, C2_Raw)