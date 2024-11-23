import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if dropout_rate > 0:
            self.double_conv.add_module("dropout", nn.Dropout2d(dropout_rate))

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=32):
        super(UNet, self).__init__()
        self.encoder1 = DoubleConv(in_channels, num_features)
        self.encoder2 = DoubleConv(num_features, num_features*2)
        self.encoder3 = DoubleConv(num_features*2, num_features*4)
        self.encoder4 = DoubleConv(num_features*4, num_features*8)
        self.encoder5 = DoubleConv(num_features*8, num_features*16)

        self.decoder1 = nn.ConvTranspose2d(num_features*16, num_features*8, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(num_features*8, num_features*4, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose2d(num_features*4, num_features*2, kernel_size=2, stride=2)
        self.decoder4 = nn.ConvTranspose2d(num_features*2, num_features, kernel_size=2, stride=2)
        
        self.up1 = DoubleConv(num_features*16, num_features*8)
        self.up2 = DoubleConv(num_features*8, num_features*4)
        self.up3 = DoubleConv(num_features*4, num_features*2)
        self.up4 = DoubleConv(num_features*2, num_features)

        self.out = nn.Conv2d(num_features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        enc5 = self.encoder5(F.max_pool2d(enc4, 2))

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
