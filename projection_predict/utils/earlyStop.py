import torch
import os
from torch import nn
import wandb

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt', verbose=True):
        """
        :param patience: 等待验证集性能改善的最大轮数
        :param delta: 最小变化量（用于确定是否算作性能改善）
        :param path: 检查点保存路径
        :param verbose: 是否打印提示信息
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, exp_id, flag):
        score = -val_loss  # 以负验证损失作为性能衡量标准

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, exp_id, flag)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Early stop counting: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, exp_id, flag)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, exp_id, flag):
        os.makedirs('./checkpoints', exist_ok=True)
        if self.verbose:
            print(f"Val loss decrease: ({self.val_loss_min:.6f} --> {val_loss:.6f}), Saving checkpoint.")
        if isinstance(model, nn.DataParallel):
            # Save the model without 'module.' prefix
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        torch.save({'weight': model_state_dict},
                   f'./checkpoints/{exp_id}_{flag}.pth')
        wandb.save(f'./checkpoints/{exp_id}_{flag}.pth', 
                   base_path='./checkpoints',
                   policy='live'
                   )
        self.val_loss_min = val_loss