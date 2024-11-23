import numpy as np
import torch
import os

def main():

    filename = '/root/autodl-tmp/denoise/data/Val/Val_LIDC_128_transposed.npz'
    voxel_normalize(filename)

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
    i = 0
    for index in range(len(data)):
        judge = np.max(data[index])
        if int(judge) != 1:
            data[index] = data[index] / judge
            i +=1
    print(i)
    np.savez('/root/autodl-tmp/denoise/data/Val/Val_LIDC_128.npz', data)

if __name__ == '__main__':
    main()