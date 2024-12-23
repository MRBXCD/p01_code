import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channel):
        self.in_channel = in_channel
        super(UNet3D, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, bias=False, batchnorm=False)
        self.ec1 = self.encoder(32, 64, bias=False, batchnorm=False)
        self.ec2 = self.encoder(64, 64, bias=False, batchnorm=False)
        self.ec3 = self.encoder(64, 128, bias=False, batchnorm=False)
        self.ec4 = self.encoder(128, 128, bias=False, batchnorm=False)
        self.ec5 = self.encoder(128, 256, bias=False, batchnorm=False)
        self.ec6 = self.encoder(256, 256, bias=False, batchnorm=False)
        self.ec7 = self.encoder(256, 512, bias=False, batchnorm=False)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder(64, self.in_channel, kernel_size=5, stride=1, bias=False)

        #self.dropout = nn.Dropout3d(p=0.5)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        #print("e0 size:", e0.size())
        #e0 = self.dropout(e0)
        syn0 = self.ec1(e0)

        #syn0 = self.dropout(syn0)

        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)

        #e2 = self.dropout(e2)

        syn1 = self.ec3(e2)

        #syn1 = self.dropout(syn1)

        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)

        #e4 = self.dropout(e4)

        syn2 = self.ec5(e4)

        #syn2 = self.dropout(syn2)
        del e3, e4

        e5 = self.pool2(syn2)
        # print(f'syn2 size {syn2.size()}, e5 size {e5.size()}')
        e6 = self.ec6(e5)

        #e6 = self.dropout(e6)

        e7 = self.ec7(e6)

        #e7 = self.dropout(e7)
        #print("e7 size:", e7.size())
        del e5, e6

        t9 = self.dc9(e7)
        #t9 = self.dropout(t9)
        #print('t9 size', t9.size())
        t9_upSampled = F.interpolate(t9, size=syn2.size()[2:], mode='trilinear', align_corners=False)
        #print("t9 size:", t9_upSampled.size(), "syn2 size:", syn2.size())
        d9 = torch.cat((t9_upSampled, syn2), dim=1)
        #print('d9 size:', d9.size())
        del e7, syn2, t9_upSampled

        d8 = self.dc8(d9)
        #d8 = self.dropout(d8)
        d7 = self.dc7(d8)
        #d7 = self.dropout(d7)
        del d9, d8

        t6 = self.dc6(d7)
        #t6 = self.dropout(t6)
        #print('t6 size', t6.size())
        t6_upSampled = F.interpolate(t6, size=syn1.size()[2:], mode='trilinear', align_corners=False)
        #print("t6_upSampled size:", t6_upSampled.size(), "syn1 size:", syn1.size())
        d6 = torch.cat((t6_upSampled, syn1), dim=1)
        del d7, syn1, t6_upSampled

        d5 = self.dc5(d6)
        #d5 = self.dropout(d5)
        d4 = self.dc4(d5)
        #d4 = self.dropout(d4)
        del d6, d5

        t3 = self.dc3(d4)
        #t3 = self.dropout(t3)
        t3_upSampled = F.interpolate(t3, size=syn0.size()[2:], mode='trilinear', align_corners=False)
        d3 = torch.cat((t3_upSampled, syn0), dim=1)
        del d4, syn0, t3_upSampled

        d2 = self.dc2(d3)
        #d2 = self.dropout(d2)
        d1 = self.dc1(d2)
        #d1 = self.dropout(d1)
        del d3, d2

        d0 = self.dc0(d1)
        #print('size of output:', d0.size())
        return d0
