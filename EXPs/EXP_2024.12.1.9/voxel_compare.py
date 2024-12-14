import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse

voxel_16_dlR = np.load('/root/autodl-tmp/p01_code/reconstruction/result/voxel/Val_Recons_SIRT_16_angles.npz')['arr_0']
voxel_16_rawR = np.load('/root/autodl-tmp/shared_data/voxel/recons/raw/Val_Recons_SIRT_8_angles.npz')['arr_0']
voxel_Raw = np.load('/root/autodl-tmp/shared_data/voxel/raw/Val_LIDC_128.npz')['arr_0']

voxel_16_rawR = voxel_16_rawR.flatten()
voxel_16_dlR = voxel_16_dlR.flatten()
voxel_Raw = voxel_Raw.flatten()
rr_Raw = rmse(voxel_16_rawR, voxel_Raw)
dlr_Raw = rmse(voxel_16_dlR, voxel_Raw)

print(dlr_Raw,rr_Raw)