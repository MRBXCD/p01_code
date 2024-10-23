import numpy as np
import matplotlib.pyplot as plt

datain = np.load('/root/autodl-tmp/Projection_predict/result/inference/8-16_model_input_train.npz')['arr_0']
datatar = np.load('/root/autodl-tmp/Projection_predict/result/inference/8-16_model_input_train.npz')['arr_0']

fig, axi = plt.subplot()