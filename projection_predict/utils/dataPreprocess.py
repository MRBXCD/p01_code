import numpy as np
import torch
import os

def main():
    folder = './data'
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        voxel_normalize(filepath)

def voxel_squeeze(filename):
    data = np.load(filename)['arr_0']
    print(data.shape)
    if data.ndim == 4:
        print(f'{filename} already squeezed')
    else:
        squeezed_data = np.squeeze(data, axis=None)
        np.savez(filename, squeezed_data)
        print(f'{filename} squeeze complete')

def voxel_transpose(data):
    transposed_data = []
    for index in range(len(data)):
        transposed_data.append(np.transpose(data[index], (2, 1, 0)))
    np.savez('data/Val_LIDC_128_transposed.npz', transposed_data)
    print('end')

def voxel_normalize(filename):
    data = np.load(filename)['arr_0']
    judge = np.max(data)
    print(f'The max value of {filename} is {judge}')
    if int(judge) != 1:
        data_normalized = data / judge
        np.savez(filename, data_normalized)
        print(f'{filename} normalize complete')
    else:
        print(f'{filename} already normalized')

if __name__ == '__main__':
    main()