import numpy as np
import matplotlib.pyplot as plt


data_interpoler = np.load('/root/autodl-tmp/Projection_predict/8-16_projections_train.npz')['arr_0']

proj = data_interpoler[0]

for index in range(16):
    plt.imsave(f'./test/{index}.png', proj[:,index,:], cmap='gray')
    