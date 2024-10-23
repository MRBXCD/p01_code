import numpy as np
import matplotlib.pyplot as plt
import os

views = '32val'
voxel_data = np.load("/root/autodl-tmp/denoise/data/Val/Val_Recons_SIRT_32_angles.npz")['arr_0']
voxel_data = voxel_data.squeeze()
print(np.shape(voxel_data))

# Randomly choose 10 indeices for further illustration
index = [58, 79, 62, 26, 84]
index = [8, 5, 63, 18, 41]
os.makedirs(f'/root/autodl-tmp/paper_revise/images/{views}', exist_ok=True)
samples = voxel_data[index,:,:,:]

num = 0
for sample in samples:
    slice = sample[64,:,:]
    plt.imsave(f'/root/autodl-tmp/paper_revise/images/{views}/{num}_slice.png', slice, cmap='gray')
    num += 1

print('Image')

 