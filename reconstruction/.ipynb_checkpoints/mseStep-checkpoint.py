import numpy as np
import torch
import matplotlib.pyplot as plt


ANGLES = 16
TARGET = np.load('/root/autodl-tmp/Reconstruction/result/voxel/Train_Recons_SIRT_16_angles.npz')['arr_0']
INPUT = np.load('./data/voxel/Train_LIDC_128.npz')['arr_0']
print(np.shape(TARGET))
TARGET_t = torch.tensor(TARGET)
INPUT_t = torch.tensor(INPUT)

if np.max(TARGET) == 1:
    print('TARGET data checked')

if np.max(INPUT) == 1:
    print('INPUT data checked')

# voxel = TARGET[0]
# for index in range(128):
#     plt.imsave(f'./test/{index}.png', voxel[:,index,:], cmap='gray')


# error = []
# for index in range(66):
#     error.append(TARGET[index,:,:,:] - INPUT[index,:,:,:])
# array = np.array(error)
# square = array ** 2
# print(np.shape(array))
# mse = np.mean(square)

loss_fn = torch.nn.MSELoss()
error = loss_fn(INPUT_t, TARGET_t)

print(f'Loss between {ANGLES}-Raw is {error.item()}')