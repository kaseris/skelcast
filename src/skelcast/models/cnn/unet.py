import torch
import torch.nn as nn

from skelcast.models import MODELS
from skelcast.models.module import SkelcastModule


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding='same', padding_mode='reflect', bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    
class UpConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, mode='bilinear'):
        super().__init__()
        self.us = nn.Upsample(scale_factor=1, mode=mode)
        self.conv = Conv2D(in_channels, out_channels, kernel, stride=1)

    def forward(self, x):
        x = self.us(x)
        x = self.conv(x)
        return x
    

class DownConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels, kernel, stride=1)

    def forward(self, x):
        return self.conv(x)
    

class CatConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1):
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels, kernel, stride=1)

    def forward(self, x1, x2):
        return self.conv(torch.cat([x1, x2], dim=1))
    
    
@MODELS.register_module()
class Unet(SkelcastModule):
    def __init__(self, filters=64, seq_size=50, out_size=5, loss_fn: nn.Module = None):
        super().__init__()
        # Decoder
        self.c1 = Conv2D(seq_size, filters, 1)
        self.c2 = DownConv2D(filters, filters * 2, 1)
        self.c3 = DownConv2D(filters * 2, filters * 4, 1)
        self.c4 = DownConv2D(filters * 4, filters * 8, 1)
        # Bottleneck
        self.c5 = DownConv2D(filters * 8, filters * 8, 1)
        # Encoder
        self.u1 = UpConv2D(filters * 8, filters * 8, 1, mode='bilinear')
        self.cc1 = CatConv2D(filters * 16, filters * 8, 1)
        self.u2 = UpConv2D(filters * 8, filters * 4, 1, mode='bilinear')
        self.cc2 = CatConv2D(filters * 8, filters * 4, 1)
        self.u3 = UpConv2D(filters * 4, filters * 2, 1, mode='bilinear')
        self.cc3 = CatConv2D(filters * 4, filters * 2, 1)
        self.u4 = UpConv2D(filters * 2, filters, 1, mode='bilinear')
        self.cc4 = CatConv2D(filters * 2, filters, 1)
        self.outconv = Conv2D(filters, out_size, 1)  

        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()  

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.c5(x4)
        x = self.u1(x5)
        x = self.cc1(x, x4)
        x = self.u2(x)
        x = self.cc2(x, x3)
        x = self.u3(x)
        x = self.cc3(x, x2)
        x = self.u4(x)
        x = self.cc4(x, x1)
        x = self.outconv(x)
        return x
    
    def training_step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        out = self(x)
        loss = self.loss_fn(out, y)
        return {'out': out, 'loss': loss}
    
    @torch.no_grad()
    def validation_step(self, x, y) -> dict:
        out = self(x)
        loss = self.loss_fn(out, y)
        return {'out': out, 'loss': loss}
    
    @torch.no_grad()
    def predict(self):
        pass