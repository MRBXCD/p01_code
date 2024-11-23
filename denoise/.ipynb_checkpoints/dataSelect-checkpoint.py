import random
import numpy as np

angles = 16

random_integers = [random.randint(0, 766) for _ in range(100)]
print(random_integers)

data_input = np.load(f'./data/Train/Train_Recons_16angles_1000.npz')['arr_0']
data_target = np.load('./data/Train/Train_LIDC_128_transposed.npz')['arr_0']

selected_data_input = data_input[random_integers]
selected_data_target = data_target[random_integers]
np.savez(f'./data/Input_LIDC_128_small_{angles}.npz', selected_data_input)
np.savez(f'./data/Target_LIDC_128_small_Raw.npz', selected_data_target)