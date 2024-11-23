import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse

B = np.load('/home/mrb2/experiments/graduation_project/p01_code/EXP_2024.10.1.3/projections/Projection_val_data_16_angles_padded.npz')['arr_0']
B2 = np.load('/home/mrb2/experiments/graduation_project/p01_code/EXP_2024.10.1.3/val.npz')['arr_0']
D = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/dl/projections_val_8-16.npz')['arr_0']
Raw = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/raw/Projection_val_data_16_angles_padded.npz')['arr_0']
B2_Raw = rmse(B2.flatten(), Raw.flatten())
D_Raw = rmse(D.flatten(), Raw.flatten())

B2_image = B2[0].reshape(148,16*148)
D_image = D[0].reshape(148,16*148)
Raw_image = Raw[0].reshape(148,16*148)


fig, axes = plt.subplots(3, 1, figsize=(10, 4))

axes[0].imshow(B2_image, cmap='gray')
axes[0].set_title(f'B2, B2_Raw:{B2_Raw}')
axes[1].imshow(D_image, cmap='gray')
axes[1].set_title(f'D, D_Raw:{D_Raw}')
axes[2].imshow(Raw_image, cmap='gray')
axes[2].set_title(f'Raw')
    
# plt.tight_layout()
plt.savefig('/home/mrb2/experiments/graduation_project/p01_code/EXP_2024.10.1.3/EXP_4_projection_image.png', dpi=500)
plt.close()

B_train = np.load('/home/mrb2/experiments/graduation_project/p01_code/EXP_2024.10.1.3/projections/Projection_train_data_16_angles_padded.npz')['arr_0']
B2_train = np.load('/home/mrb2/experiments/graduation_project/p01_code/EXP_2024.10.1.3/train.npz')['arr_0']
D_train = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/dl/projections_train_8-16.npz')['arr_0']
Raw_train = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/raw/Projection_train_data_16_angles_padded.npz')['arr_0']

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
plt.plot(B2_Raw_list_train, label='B2_Raw', alpha=1)
plt.plot(D_Raw_list_train, label='D_Raw', alpha=1)
plt.plot(B_Raw_list_train, label='B_Raw', alpha=1)
plt.legend()
plt.grid(True)
plt.savefig('/home/mrb2/experiments/graduation_project/p01_code/EXP_2024.10.1.3/EXP_4_rmse_train.png')

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
plt.plot(B2_Raw_list_val, label='B2_Raw', alpha=1)
plt.plot(D_Raw_list_val, label='D_Raw', alpha=1)
plt.plot(B_Raw_list_val, label='B_Raw', alpha=1)
plt.legend()
plt.grid(True)
plt.savefig('/home/mrb2/experiments/graduation_project/p01_code/EXP_2024.10.1.3/EXP_4_rmse_val.png')