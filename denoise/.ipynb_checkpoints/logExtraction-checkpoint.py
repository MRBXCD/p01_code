import torch
import numpy as np
import matplotlib.pyplot as plt
import os

STAGE = '64-360'
STAGE1 = '64-Raw'
NUM_GPU = 2
CHECK_POINT = 200
PRE_INFOR = torch.load(f'./weight/{NUM_GPU}_GPU/{STAGE}/model_checkpoint_{CHECK_POINT}_epoch.pth')

TRAIN_LOSSES = np.array(PRE_INFOR['losses_train']) 
VAL_LOSSES = np.array(PRE_INFOR['losses_val']) 
upbound = np.max(TRAIN_LOSSES[:,2])
EPOCH = PRE_INFOR['epoch']
EPOCH_VECTOR = list(range(0, EPOCH+1))


os.makedirs('./loss_figure', exist_ok=True)
fig, ax = plt.subplots()

# 设置坐标轴的背景颜色为浅蓝色
ax.set_facecolor('lightgray')
ax.plot(EPOCH_VECTOR, TRAIN_LOSSES[:,2],  linestyle='-')
ax.plot(EPOCH_VECTOR, VAL_LOSSES[:,2],  linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('MSE LOSS')
plt.ylim(0,upbound)
plt.title(f'Loss Under {STAGE1} stage')
plt.grid(True)
plt.savefig(f'./loss_figure/Loss_Under_{STAGE1}_stage.png')