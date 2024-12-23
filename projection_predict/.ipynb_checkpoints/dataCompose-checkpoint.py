import numpy as np
import matplotlib.pyplot as plt

# raw is odd index
data_raw_train = np.load('/root/autodl-tmp/Projection_predict/result/inference/16-1_model_input_train.npz')['arr_0']
data_raw_train = data_raw_train.reshape(767, 148, 8, 148)
print(np.shape(data_raw_train))

# predict is even index
data_predict_train = np.load('/root/autodl-tmp/Projection_predict/result/inference/16-1_model_output_train.npz')['arr_0']
data_predict_train = data_predict_train.reshape(767, 148, 8, 148)
print(np.shape(data_predict_train))

data_raw_val = np.load('/root/autodl-tmp/Projection_predict/result/inference/16-1_model_input_val.npz')['arr_0']
data_raw_val = data_raw_val.reshape(66, 148, 8, 148)
print(np.shape(data_raw_val))

# predict is even index
data_predict_val = np.load('/root/autodl-tmp/Projection_predict/result/inference/16-1_model_output_val.npz')['arr_0']
data_predict_val = data_predict_val.reshape(66, 148, 8, 148)
print(np.shape(data_predict_val))

fig, axs = plt.subplots(2, 8)
plt.subplots_adjust(wspace=0.1, hspace=0)
indices = 0
result_train = []

for index in range(len(data_predict_train)):
    composed_data = []
    for i in range(8):
        even_proj = data_raw_train[index,:,i,:]
        odd_proj = data_predict_train[index,:,i,:]
        composed_data.append(even_proj)
        composed_data.append(odd_proj)
    composed_data = np.transpose(composed_data, (1,0,2))    
    result_train.append(composed_data)

for index in range(16):
    plt.imsave(f'./test/{index}.png', result_train[1][:,index,:], cmap='gray')

    
result_val = []    
for index in range(len(data_predict_val)):
    composed_data = []
    for i in range(8):
        even_proj = data_raw_val[index,:,i,:]
        odd_proj = data_predict_val[index,:,i,:]
        composed_data.append(even_proj)
        composed_data.append(odd_proj)
    composed_data = np.transpose(composed_data, (1,0,2))    
    result_val.append(composed_data)    
print(np.shape(result_val))

# for index in range(16):
#     plt.imsave(f'./test/{index}.png', composed_data[:,index,:],cmap='gray')
# #for index in range(8):
np.savez('8-16_projections_train.npz', result_train)
np.savez('8-16_projections_val.npz', result_val)


