import torch
import torch.nn as nn
from lightning import LightningModule
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None, residual = False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1, bias = False), 
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.GroupNorm(1, out_channels),
        )
        
    def forward(self, x):
        
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual = True),
            DoubleConv(in_channels, out_channels, residual = False)
        )
        
    def forward(self, x):        
        return self.max_pool(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True)
            self.conv = DoubleConv(in_channels, in_channels, residual = True)
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 2, stride = 2)
            self.conv = DoubleConv(in_channels, in_channels, residual = True)
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        ## N, C, H, W
        diffX = x2.size()[2] - x1.size()[2] ## H
        diffY = x2.size()[3] - x1.size()[3] ## W
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim = 1)
        x = self.conv(x)
        x = self.conv2(x)
        
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x):
        return self.conv(x)
        
        


class AttentionUnet(nn.Module):
    def __init__(self, img_depth, device):
        
        super().__init__()
        
        if device == "gpu":
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.img_depth = img_depth

        self.inc = DoubleConv(img_depth, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        bilinear = True
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.out = OutConv(64, img_depth)
        
        self.factor = factor
        
    def positional_encoding(self, t, channels, embed_size):
        inv_freq = 1 / (10000 ** (torch.arange(0, channels, 2, device = self.device).float()) / channels)## channels/2
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq) ## batch_size, channels // 2
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq) ## batch_size, channels // 2
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim = -1) ## batch_size, channels
        pos_enc = pos_enc.view(-1, channels, 1, 1)
        
        return pos_enc.repeat(1, 1, embed_size, embed_size) ## batch_size, channels, embed_size, embed_size
        
    
    def forward(self, x, t):
        x1 = self.inc(x)
        # print(x1.shape)
        # print(self.down1(x1).shape)
        x2 = self.down1(x1) + self.positional_encoding(t, 128, 16)
        x3 = self.down2(x2) + self.positional_encoding(t, 256, 8)
        x4 = self.down3(x3) + self.positional_encoding(t, 512 // self.factor, 4)
        
        x = self.up1(x4, x3) + self.positional_encoding(t, 256 // self.factor, 8)

        x = self.up2(x, x2) + self.positional_encoding(t, 128 // self.factor, 16)
        x = self.up3(x, x1) + self.positional_encoding(t, 64, 32)
        
        output = self.out(x)
        
        return output

def main():
    
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Select the CUDA device
    
    model = AttentionUnet(1, "gpu").to(device)
    # print(model)
    x = torch.randn(3, 1, 32, 32).to(device) ## B, C, H, W
    print(model.forward(x, torch.Tensor([999, 999, 999]).unsqueeze(-1).to(device)))

if __name__ == "__main__":
    main()
        
        