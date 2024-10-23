import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

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

class UNetMultiPath(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=32, paths=3):
        super(UNetMultiPath, self).__init__()
        print('MODEL LOAD FROM -model.py-: UNetMultiPath ')
        self.paths = paths
        
        # Creating lists for encoders and decoders for each path
        self.encoders = nn.ModuleList([
            nn.ModuleList([
                DoubleConv(in_channels if i == 0 else num_features * (2 ** i), num_features * (2 ** i)) 
                for i in range(5)
            ]) for _ in range(paths)
        ])
        
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                nn.ConvTranspose2d(num_features * (2 ** (4-i)), num_features * (2 ** (3-i)), kernel_size=2, stride=2)
                for i in range(4)
            ]) for _ in range(paths)
        ])
        
        self.ups = nn.ModuleList([
            nn.ModuleList([
                DoubleConv(num_features * (2 ** (4-i)) * 2, num_features * (2 ** (3-i)))
                for i in range(4)
            ]) for _ in range(paths)
        ])

        # One output convolution for each path
        self.out_convs = nn.ModuleList([nn.Conv2d(num_features, out_channels, kernel_size=1) for _ in range(paths)])

    def forward(self, x):
        outputs = []

        for path in range(self.paths):
            # Encoder path
            encs = []
            for i, encoder in enumerate(self.encoders[path]):
                x = encoder(x if i == 0 else F.max_pool2d(encs[i-1], 2))
                encs.append(x)

            # Decoder path
            for i, (decoder, up) in enumerate(zip(self.decoders[path], self.ups[path])):
                x = decoder(encs[-(i+2)]) # Get the output of the corresponding encoder layer
                x = self.crop_and_concat(encs[-(i+2)], x)
                x = up(x)

            # Output layer
            outputs.append(self.out_convs[path](x))

        return outputs

    def crop_and_concat(self, enc_tensor, dec_tensor):
        _, _, H, W = enc_tensor.size()
        dec_tensor = F.interpolate(dec_tensor, size=(H, W), mode='bilinear', align_corners=True)
        return torch.cat((enc_tensor, dec_tensor), dim=1)