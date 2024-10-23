import numpy as np
import torch

data_halfway = np.load('/root/autodl-tmp/Reconstruction/result/projection/64_angles/Projection_train_data_64_angles_padded.npz')['arr_0']
data_stright = np.load('/root/autodl-tmp/Projection_predict/data/train/Projection_train_data_64_angles_padded.npz')['arr_0']
print(np.shape(data_halfway), np.max(data_halfway))
print(np.shape(data_stright), np.max(data_stright))
lossF = torch.nn.MSELoss()

if np.array_equal(data_halfway, data_stright):
    print('same')
else:
    print('not same')