import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=32, dropout_rate=0):
        super(UNet, self).__init__()
        # 编码器（下采样路径）
        self.encoder1 = DoubleConv(in_channels, num_features)
        self.encoder2 = DoubleConv(num_features, num_features*2)
        self.encoder3 = DoubleConv(num_features*2, num_features*4)
        self.encoder4 = DoubleConv(num_features*4, num_features*8)
        self.encoder5 = DoubleConv(num_features*8, num_features*16)

        # 解码器（上采样路径）
        self.decoder1 = nn.ConvTranspose2d(num_features*16, num_features*8, kernel_size=2, stride=2)
        self.up1 = DoubleConv(num_features*16, num_features*8, dropout_rate)

        self.decoder2 = nn.ConvTranspose2d(num_features*8, num_features*4, kernel_size=2, stride=2)
        self.up2 = DoubleConv(num_features*8, num_features*4, dropout_rate)

        self.decoder3 = nn.ConvTranspose2d(num_features*4, num_features*2, kernel_size=2, stride=2)
        self.up3 = DoubleConv(num_features*4, num_features*2, dropout_rate)

        self.decoder4 = nn.ConvTranspose2d(num_features*2, num_features, kernel_size=2, stride=2)
        self.up4 = DoubleConv(num_features*2, num_features, dropout_rate)

        self.out = nn.Conv2d(num_features, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        enc5 = self.encoder5(F.max_pool2d(enc4, 2))

        # 解码器
        dec1 = self.decoder1(enc5)
        dec1 = self.crop_and_concat(enc4, dec1)
        dec1 = self.up1(dec1)

        dec2 = self.decoder2(dec1)
        dec2 = self.crop_and_concat(enc3, dec2)
        dec2 = self.up2(dec2)

        dec3 = self.decoder3(dec2)
        dec3 = self.crop_and_concat(enc2, dec3)
        dec3 = self.up3(dec3)

        dec4 = self.decoder4(dec3)
        dec4 = self.crop_and_concat(enc1, dec4)
        dec4 = self.up4(dec4)

        out = self.out(dec4)
        return out

    def crop_and_concat(self, enc_tensor, dec_tensor):
        _, _, H, W = enc_tensor.size()
        dec_tensor = F.interpolate(dec_tensor, size=(H, W), mode='bilinear', align_corners=True)
        return torch.cat((enc_tensor, dec_tensor), dim=1)
