# libraries
import matplotlib.pyplot as plt
import numpy as np

# Create a dataset:
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file.readlines()]
    return data

file1_path = '/root/autodl-tmp/Reconstruction/result/voxel/logs/total_mse_train_SIRT_32_angles.txt'
data1 = load_data(file1_path)
print(np.var(data1))
x = range(0, len(data1))
# plot
plt.figure()
plt.plot( x, data1, linestyle='none', marker='o')
plt.ylim([0,0.005])
plt.ylabel('MSE Value (Normalized)')
plt.xlabel('Individuals')
plt.savefig('./mse.png')
plt.show()