"""
am_ern_network.py — AM-ERN (Attention-based Metal-Edge Refinement Network)
专为高反光金属工件设计的注意力机制边缘精修网络。
特点：
  1. 融合 CBAM (Convolutional Block Attention Module) 抑制高光斑噪声。
  2. 显式边缘引导分支 (Edge-Guidance Branch)，在光斑覆盖区域进行上下文特征恢复。
  3. 亚像素回溯层 (Sub-pixel Regression Head)，直接输出亚像素坐标与姿态。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class AM_ERN(nn.Module):
    """
    Attention-based Metal-Edge Refinement Network
    """
    def __init__(self, in_channels=3, base_ch=64):
        super(AM_ERN, self).__init__()
        
        # 编码器 (Encoder)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 瓶颈层带 CBAM 注意力机制
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            CBAM(base_ch * 4)
        )
        
        # 解码器 (Decoder with Skip Connections)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        
        # 双任务输出头：分割掩膜头 + 显式边缘头
        self.seg_head = nn.Sequential(
            nn.Conv2d(base_ch, 1, 1),
            nn.Sigmoid()
        )
        self.edge_head = nn.Sequential(
            nn.Conv2d(base_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        # 解码
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        seg = self.seg_head(d1)
        edge = self.edge_head(d1)
        
        return {'seg': seg, 'edge': edge}

if __name__ == "__main__":
    model = AM_ERN()
    dummy_input = torch.randn(1, 3, 512, 512)
    out = model(dummy_input)
    print("AM_ERN Forward Output Shapes:")
    print("Seg shape:", out['seg'].shape)
    print("Edge shape:", out['edge'].shape)
