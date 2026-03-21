import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from motion_magnification_learning_based_master.magnet import Manipulator as MagManipulator
from motion_magnification_learning_based_master.magnet import Encoder_No_texture as MagEncoder_No_texture

# =========================
# ECA Attention
# =========================
class ECA(nn.Module):
    def __init__(self, channel):
        super().__init__()
        t = int(abs(math.log(channel, 2) + 1) / 2)
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=(k-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y

# =========================
# CBAM Attention
# =========================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # channel attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()
        # spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # channel attention
        avg_pool = x.mean((2,3))
        max_pool,_ = x.max((2,3))
        ca = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        x = x * ca.unsqueeze(-1).unsqueeze(-1)
        # spatial attention
        avg_out = x.mean(dim=1, keepdim=True)
        max_out,_ = x.max(dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.sigmoid(self.conv_spatial(sa))
        return x * sa

# =========================
# Multi-Scale Convolution
# =========================
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, 7, padding=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv3(x) + self.conv5(x) + self.conv7(x)
        out = self.bn(out)
        return self.relu(out)

# =========================
# Input Enhancement
# =========================
class FlowAug(nn.Module):
    def forward(self, x):
        noise = (torch.rand_like(x) - 0.5) * 0.1
        return x + noise

# =========================
# Cross-Branch Attention (with non-linear fusion)
# =========================
class CrossBranchAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim*3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x1, x2, x3):
        q = self.q(x1)
        k = self.k(x3)
        v = self.v(x2)
        attn = torch.matmul(q, k.T) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        fused = torch.cat([x1, out, x3], dim=1)
        return self.fusion_mlp(fused)

# =========================
# Multi-Scale Temporal Attention
# =========================
class MultiScaleTemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_short = nn.MultiheadAttention(embed_dim=dim, num_heads=4)
        self.attn_long = nn.MultiheadAttention(embed_dim=dim, num_heads=4)
    def forward(self, x):
        x_short = x[:, :2, :]
        x_long = x[:, :, :]
        out_short,_ = self.attn_short(x_short.permute(1,0,2), x_short.permute(1,0,2), x_short.permute(1,0,2))
        out_long,_ = self.attn_long(x_long.permute(1,0,2), x_long.permute(1,0,2), x_long.permute(1,0,2))
        out = torch.cat([out_short.mean(0), out_long.mean(0)], dim=1)
        return out

# =========================
# Main Model
# =========================
class SKD_TSTSAN(nn.Module):
    def __init__(self, num_classes=5, amp_factor=5, prototypes_per_class=3):
        super().__init__()
        self.amp_factor = amp_factor
        self.enc_L = MagEncoder_No_texture(16)
        self.enc_S = MagEncoder_No_texture(1)
        self.enc_T = MagEncoder_No_texture(2)
        self.man_L = MagManipulator()
        self.man_S = MagManipulator()
        self.man_T = MagManipulator()

        # input enhancement
        self.augment = FlowAug()

        # multi-scale conv
        self.conv1_L = MultiScaleConv(32, 64)
        self.conv1_S = MultiScaleConv(32, 64)
        self.conv1_T = MultiScaleConv(32, 64)
        self.pool = nn.MaxPool2d(5,2,2)
        self.eca = ECA(64)
        self.cbam = CBAM(64)

        # deeper conv
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # final conv
        self.conv4 = nn.Conv2d(64,128,3,padding=1)
        self.conv5 = nn.Conv2d(128,128,3,padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # attention
        self.temporal_attn = MultiScaleTemporalAttention(128)
        self.cross_attn = CrossBranchAttention(128)

        # classification
        self.fc = nn.Linear(128*2, num_classes) # after temporal concatenation
        self.fc_ac1 = nn.Linear(128, num_classes)
        self.fc_ac2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

        # prototype (multi per class)
        self.prototypes_per_class = prototypes_per_class
        self.prototypes = nn.Parameter(torch.randn(num_classes, prototypes_per_class, 128))
        self.proj = nn.Linear(128*2,128)
        self.register_buffer('proto_ema', self.prototypes.data.clone())

    def forward(self, input):
        # ===== split =====
        x1 = input[:,2:18]
        x1_on = input[:,18:34]
        x2 = input[:,0:1]
        x2_on = input[:,1:2]
        x3 = input[:,34:]
        b = x1.size(0)

        # augment input
        x1 = self.augment(x1)
        x2 = self.augment(x2)
        x3 = self.augment(x3)

        # motion magnification
        x1 = self.man_L(self.enc_L(x1_on), self.enc_L(x1), self.amp_factor)
        x2 = self.man_S(self.enc_S(x2_on), self.enc_S(x2), self.amp_factor)
        x3 = self.man_T(self.enc_T(torch.zeros_like(x3)), self.enc_T(x3), self.amp_factor)

        # shallow conv
        x1 = self.pool(self.cbam(self.conv1_L(x1)))
        x2 = self.pool(self.cbam(self.conv1_S(x2)))
        x3 = self.pool(self.cbam(self.conv1_T(x3)))
        x1 = self.eca(x1)

        # deeper conv
        x1 = self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(x1))))))
        x2 = self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(x2))))))
        x3 = self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(x3))))))

        # final conv
        x1 = self.global_pool(self.relu(self.bn5(self.conv5(self.relu(self.bn4(self.conv4(x1)))))))
        x2 = self.global_pool(self.relu(self.bn5(self.conv5(self.relu(self.bn4(self.conv4(x2)))))))
        x3 = self.global_pool(self.relu(self.bn5(self.conv5(self.relu(self.bn4(self.conv4(x3)))))))

        x1 = x1.reshape(b,-1)
        x2 = x2.reshape(b,-1)
        x3 = x3.reshape(b,2,-1)
        x3 = self.temporal_attn(x3)
        x3 = x3.mean(dim=1)

        # cross-branch fusion
        final_feature = self.cross_attn(x1,x2,x3)
        final_feature = self.dropout(final_feature)
        final_feature_proj = self.proj(final_feature)

        # AC1/AC2
        AC1_out = self.fc_ac1(x1)
        AC2_out = self.fc_ac2(x2)

        # final classification
        yhat = self.fc(torch.cat([final_feature, x3], dim=1))
        return yhat, AC1_out, AC2_out, final_feature, final_feature_proj, x1, x2

    @torch.no_grad()
    def update_prototypes(self, features, labels, momentum=0.9):
        # multi-prototype update
        for i in range(self.prototypes.size(0)):
            mask = labels==i
            if mask.sum()==0:
                continue
            feat_mean = features[mask].mean(dim=0)
            self.proto_ema[i] = momentum*self.proto_ema[i]+(1-momentum)*feat_mean
            self.prototypes.data[i] = self.proto_ema[i]

# =========================
# get_model
# =========================
def get_model(model_name, class_num, alpha):
    """
    不需要 alpha，因为融合权重已经变成可学习的了。
    """
    if model_name=="SKD_TSTSAN":
        return SKD_TSTSAN(class_num)