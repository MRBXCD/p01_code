import torch
import numpy as np
import matplotlib.pyplot as plt
import os
#3_Individual_Unet
STAGE = '16-1'
NUM_GPU = 1
CHECK_POINT = 250
PRE_INFOR = torch.load(f'./weight/{NUM_GPU}_GPU/{STAGE}/model_checkpoint_{CHECK_POINT}_epoch.pth')

TRAIN_LOSSES = np.array(PRE_INFOR['losses_train']) 
VAL_LOSSES = np.array(PRE_INFOR['losses_val']) 
weight = np.array(PRE_INFOR['weight']) 
# print(weight)
print(np.shape(TRAIN_LOSSES))
# upbound = np.max(TRAIN_LOSSES[:,0])
upbound = np.max(TRAIN_LOSSES)
EPOCH = PRE_INFOR['epoch']
EPOCH_VECTOR = list(range(0, EPOCH+1))


os.makedirs('./loss_figure', exist_ok=True)
plt.figure()
# plt.plot(EPOCH_VECTOR, TRAIN_LOSSES[:,0], marker='o', linestyle='-')
# plt.plot(EPOCH_VECTOR, VAL_LOSSES[:,0], marker='o', linestyle='-')
plt.plot(EPOCH_VECTOR, TRAIN_LOSSES, marker='o', linestyle='-')
plt.plot(EPOCH_VECTOR, VAL_LOSSES, marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('MSE LOSS')
plt.ylim(0,upbound)
# plt.ylim(0,0.003)
plt.title(f'Loss Under {STAGE} stage')
plt.grid(True)
plt.savefig(f'./loss_figure/Loss_Under_{STAGE}_stage.png')


# print(f'PRESENT STAGE - {STAGE}')
# print(f"EPOCH {EPOCH+1} | train total loss is: {TRAIN_LOSSES[EPOCH,0]:.6f}, train 1 losses is: {TRAIN_LOSSES[EPOCH,1]:.6f}, train 2 losses is: {TRAIN_LOSSES[EPOCH,2]:.6f}, train 3 losses is: {TRAIN_LOSSES[EPOCH,3]:.6f}")
# print(f"          | val total loss is: {VAL_LOSSES[EPOCH,0]:.6f}, val 1 losses is: {VAL_LOSSES[EPOCH,1]:.6f}, val 2 losses is: {VAL_LOSSES[EPOCH,2]:.6f}, val 3 losses is: {VAL_LOSSES[EPOCH,3]:.6f}")

print(f'PRESENT STAGE - {STAGE}')
print(f"EPOCH {EPOCH+1} | train total loss is: {TRAIN_LOSSES[EPOCH]:.6f}")
print(f"          | val total loss is: {VAL_LOSSES[EPOCH]:.6f}")