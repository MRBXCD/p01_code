import numpy as np
import matplotlib.pyplot as plt


data1 = np.load('/root/autodl-tmp/Projection_predict/data/train/16_angles_projection_input.npz')['arr_0']
data2 = np.load('/root/autodl-tmp/Projection_predict/data/train/16_angles_projection_target.npz')['arr_0']
data1 = data1[0] 
for index in range(8):
    image = data1[:,index,:]
    data1[:,index,:] = image / np.max(image)
data2 = data2[0]
for index in range(8):
    image = data2[:,index,:]
    data2[:,index,:] = image / np.max(image)
data1 = data1 / np.max(data2)
data32 = np.load('/root/autodl-tmp/Reconstruction/result/projection/32_angles/Projection_train_data_32_angles_padded.npz')['arr_0']
data32 = data32[0]
data32_1 = data32[:,[0,4,8,12,16,20,24,28],:]
data32_2 = data32[:,[2,6,10,14,18,22,26,30],:]


if np.array_equal(data1[:,0,:], data32_1[:,0,:]):
    print('same')
else:
    print('not same')
indexx = 0
for index in range(8):
    plt.imsave(f'/root/autodl-tmp/Projection_predict/test/{indexx}.png', data32_1[:,index,:], cmap='gray')
    plt.imsave(f'/root/autodl-tmp/Projection_predict/test/{indexx+1}.png', data32_2[:,index,:], cmap='gray')
    indexx += 2