import numpy as np
import matplotlib.pyplot as plt

data = np.load('./data/projection/Projection_data_16_angles_padded.npz')['arr_0']

data_print = data[0]

for index in range(16):
    plt.imsave(f'./test/{index}.png', data_print[:,index,:], cmap='gray')

