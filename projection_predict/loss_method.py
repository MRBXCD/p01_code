import torch
from torchvision import models
from torch import nn

class PerceptualLoss(nn.Module):
    def __init__(self, pretrained_model='vgg16', selected_layers=None):
        super(PerceptualLoss, self).__init__()
        if selected_layers is None:
            selected_layers = [0, 5, 10]  # 选择VGG16的特定层
        self.selected_layers = selected_layers
        self.model = models.vgg16(pretrained=True).features
        self.model.eval()  # 评估模式
        for param in self.model.parameters():
            param.requires_grad = False  # 冻结参数

    def forward(self, y_pred, y_true):
        # 如果输入是单通道，扩展为 3 通道
        if y_pred.shape[1] == 1:
            y_pred = y_pred.repeat(1, 3, 1, 1)  # 在通道维度重复 3 次
        if y_true.shape[1] == 1:
            y_true = y_true.repeat(1, 3, 1, 1)  # 在通道维度重复 3 次
        
        pred_features = []
        true_features = []
        x_pred, x_true = y_pred, y_true
        
        # 逐层提取特征
        for i, layer in enumerate(self.model):
            x_pred = layer(x_pred)
            x_true = layer(x_true)
            if i in self.selected_layers:
                pred_features.append(x_pred)
                true_features.append(x_true)
        
        # 计算特征图之间的L2损失
        loss = 0
        for f_pred, f_true in zip(pred_features, true_features):
            loss += torch.mean((f_pred - f_true) ** 2)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_perceptual=0.1):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual

    def forward(self, y_pred, y_true):
        # 计算L1损失
        l1 = self.l1_loss(y_pred, y_true)
        # 计算感知损失
        perceptual = self.perceptual_loss(y_pred, y_true)
        # 综合损失
        total_loss = self.lambda_l1 * l1 + self.lambda_perceptual * perceptual
        return total_loss
