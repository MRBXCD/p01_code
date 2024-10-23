import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3Path(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        # For each head, create its own complete set of encoder and decoder components
        self.paths = nn.ModuleList([self.create_unet_path(n_channels, n_classes) for _ in range(3)])

    def create_unet_path(self, n_channels, n_classes):
        path = nn.ModuleDict({
            'inc': DoubleConv(n_channels, 64),
            'down1': Down(64, 128),
            'down2': Down(128, 256),
            'down3': Down(256, 512),
            'down4': Down(512, 512),
            'up1': Up(1024, 256),
            'up2': Up(512, 128),
            'up3': Up(256, 64),
            'up4': Up(128, 64),
            'outc': OutConv(64, n_classes)
        })
        return path

    def forward(self, x):
        outputs = []
        for path in self.paths:
            x1 = path['inc'](x)
            x2 = path['down1'](x1)
            x3 = path['down2'](x2)
            x4 = path['down3'](x3)
            x5 = path['down4'](x4)
            x11 = path['up1'](x5, x4)
            x12 = path['up2'](x11, x3)
            x13 = path['up3'](x12, x2)
            x14 = path['up4'](x13, x1)
            logits = path['outc'](x14)
            outputs.append(logits)
        return outputs

