import numpy as np
import matplotlib.pyplot as plt

def data_compose(sequence_length, train, val):
    print(sequence_length)

    train = np.array(train)
    val = np.array(val)
    data_raw_train = train[0].reshape(767, 148, sequence_length, 148)
    print(np.shape(data_raw_train))
    data_predict_train = train[1].reshape(767, 148, sequence_length, 148)
    print(np.shape(data_predict_train))
    data_raw_val = val[0].reshape(66, 148, sequence_length, 148)
    print(np.shape(data_raw_val))
    data_predict_val = val[1].reshape(66, 148, sequence_length, 148)
    print(np.shape(data_predict_val))

    result_train = []
    for index in range(len(data_predict_train)):
        composed_data = []
        for i in range(sequence_length):
            even_proj = data_raw_train[index,:,i,:]
            odd_proj = data_predict_train[index,:,i,:]
            composed_data.append(even_proj)
            composed_data.append(odd_proj)
        composed_data = np.transpose(composed_data, (1,0,2))    
        result_train.append(composed_data)   

    ex = result_train[0]
    ex = ex.reshape(148,sequence_length*2*148)    
    plt.imsave(f'./save_for_paper/compose_{sequence_length}-{sequence_length*2}.png', ex, cmap = 'gray')

    result_val = []    
    for index in range(len(data_predict_val)):
        composed_data = []
        for i in range(sequence_length):
            even_proj = data_raw_val[index,:,i,:]
            odd_proj = data_predict_val[index,:,i,:]
            composed_data.append(even_proj)
            composed_data.append(odd_proj)
        composed_data = np.transpose(composed_data, (1,0,2))    
        result_val.append(composed_data)    
    print(np.shape(result_val))

    np.savez(f'projections_train_{sequence_length}-{sequence_length*2}.npz', result_train)
    np.savez(f'projections_val_{sequence_length}-{sequence_length*2}.npz', result_val)