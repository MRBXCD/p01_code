import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse

B = np.load('/root/autodl-tmp/p01_code/reconstruction/result/voxel/Val_Recons_SIRT_16_angles_B.npz')['arr_0']
B2 = np.load('/root/autodl-tmp/p01_code/reconstruction/result/voxel/Val_Recons_SIRT_16_angles_B2.npz')['arr_0']
D = np.load('/root/autodl-tmp/p01_code/reconstruction/result/voxel/Val_Recons_SIRT_16_angles_D.npz')['arr_0']
Raw = np.load('/root/autodl-tmp/shared_data/voxel/raw/Val_LIDC_128.npz')['arr_0']

B_train = np.load('/root/autodl-tmp/p01_code/reconstruction/result/voxel/Train_Recons_SIRT_16_angles_B.npz')['arr_0']
B2_train = np.load('/root/autodl-tmp/p01_code/reconstruction/result/voxel/Train_Recons_SIRT_16_angles_B2.npz')['arr_0']
D_train = np.load('/root/autodl-tmp/p01_code/reconstruction/result/voxel/Train_Recons_SIRT_16_angles_D.npz')['arr_0']
Raw_train = np.load('/root/autodl-tmp/shared_data/voxel/raw/Train_LIDC_128.npz')['arr_0']

B_Raw_list_train = []
B2_Raw_list_train = []
D_Raw_list_train = []
for index in range(767):
    rootmse_1 = rmse(B2_train[index].flatten(), Raw_train[index].flatten())
    rootmse_2 = rmse(D_train[index].flatten(), Raw_train[index].flatten())
    rootmse_3 = rmse(B_train[index].flatten(), Raw_train[index].flatten())
    B2_Raw_list_train.append(rootmse_1)
    D_Raw_list_train.append(rootmse_2)
    B_Raw_list_train.append(rootmse_3)

plt.figure(figsize=(40,10),dpi=200)
plt.title('Train samples')
plt.plot(B2_Raw_list_train, label='B2_Raw',  color='red', alpha=0.8)
plt.plot(D_Raw_list_train, label='D_Raw', color='green', alpha=0.8)
plt.plot(B_Raw_list_train, label='B_Raw', color='blue', alpha=0.8)
plt.axhline(y=sum(B2_Raw_list_train)/len(B2_Raw_list_train), color='red', linestyle='--', linewidth=2, label='B2 average')
plt.axhline(y=sum(B_Raw_list_train)/len(B_Raw_list_train), color='green', linestyle='--', linewidth=2, label='B average')
plt.axhline(y=sum(D_Raw_list_train)/len(D_Raw_list_train), color='blue', linestyle='--', linewidth=2, label='D average')
plt.legend()
plt.grid(True)
plt.savefig('/root/autodl-tmp/p01_code/EXPs/EXP_2024.12.1.7/voxel_rmse_train.png')

B_Raw_list_val = []
B2_Raw_list_val = []
D_Raw_list_val = []
for index in range(66):
    rootmse_1 = rmse(B2[index].flatten(), Raw[index].flatten())
    rootmse_2 = rmse(D[index].flatten(), Raw[index].flatten())
    rootmse_3 = rmse(B[index].flatten(), Raw[index].flatten())
    B2_Raw_list_val.append(rootmse_1)
    D_Raw_list_val.append(rootmse_2)
    B_Raw_list_val.append(rootmse_3)

plt.figure(dpi=500)
plt.title('Val samples')
plt.plot(B2_Raw_list_val, label='B2_Raw', color='red', alpha=0.8)
plt.plot(D_Raw_list_val, label='D_Raw', color='green', alpha=0.8)
plt.plot(B_Raw_list_val, label='B_Raw', color='blue', alpha=0.8)
plt.axhline(y=sum(B2_Raw_list_val)/len(B2_Raw_list_val), color='red', linestyle='--', linewidth=2, label='B2 average')
plt.axhline(y=sum(B_Raw_list_val)/len(B_Raw_list_val), color='green', linestyle='--', linewidth=2, label='B average')
plt.axhline(y=sum(D_Raw_list_val)/len(D_Raw_list_val), color='blue', linestyle='--', linewidth=2, label='D average')
plt.legend()
plt.grid(True)
plt.savefig('/root/autodl-tmp/p01_code/EXPs/EXP_2024.12.1.7/voxel_rmse_val.png')