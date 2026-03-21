import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from motion_magnification_learning_based_master.magnet import Manipulator as MagManipulator
from motion_magnification_learning_based_master.magnet import Encoder_No_texture as MagEncoder_No_texture

# =========================
# ECA Attention (unchanged)
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
# Cross-Branch Attention with adaptive fusion
# =========================
class CrossBranchAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

        # adaptive weights
        self.alpha = nn.Parameter(torch.tensor(0.33))
        self.beta = nn.Parameter(torch.tensor(0.33))
        self.gamma = nn.Parameter(torch.tensor(0.34))

    def forward(self, x1, x2, x3):
        q = self.q(x1)
        k = self.k(x3)
        v = self.v(x2)

        attn = torch.matmul(q, k.T) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        fused = self.alpha * x1 + self.beta * out + self.gamma * x3
        return fused


# =========================
# Temporal Attention with learnable frame weights
# =========================
class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4)
        self.frame_weight = nn.Parameter(torch.ones(1, 2))  # assuming 2 frames for T branch

    def forward(self, x):
        # x shape: (B, T, C)
        weights = torch.softmax(self.frame_weight, dim=-1)
        x = x * weights.unsqueeze(-1)  # apply frame weights
        x = x.permute(1, 0, 2)
        out, _ = self.attn(x, x, x)
        return out.permute(1, 0, 2)


# =========================
# Main model with innovations
# =========================
class SKD_TSTSAN(nn.Module):
    def __init__(self, num_classes=5, amp_factor=5):
        super().__init__()

        self.amp_factor = amp_factor

        # Motion Magnification
        self.enc_L = MagEncoder_No_texture(16)
        self.enc_S = MagEncoder_No_texture(1)
        self.enc_T = MagEncoder_No_texture(2)

        self.man_L = MagManipulator()
        self.man_S = MagManipulator()
        self.man_T = MagManipulator()

        # Multi-scale Conv instead of single conv
        self.conv1_L = MultiScaleConv(32, 64)
        self.conv1_S = MultiScaleConv(32, 64)
        self.conv1_T = MultiScaleConv(32, 64)

        self.pool = nn.MaxPool2d(5, 2, 2)
        self.eca = ECA(64)

        # deeper
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # final conv
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # attention
        self.temporal_attn = TemporalAttention(128)
        self.cross_attn = CrossBranchAttention(128)

        # classification head
        self.fc = nn.Linear(128, num_classes)  # final fusion is 128 after adaptive fusion

        # AC1/AC2 heads
        self.fc_ac1 = nn.Linear(128, num_classes)
        self.fc_ac2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)

        # prototype
        self.prototypes = nn.Parameter(torch.randn(num_classes, 128))
        self.proj = nn.Linear(128, 128)  # projection for feature distillation
        self.register_buffer('proto_ema', self.prototypes.data.clone())  # EMA for prototype

    def forward(self, input):
        # ===== split =====
        x1 = input[:, 2:18]
        x1_on = input[:, 18:34]
        x2 = input[:, 0:1]
        x2_on = input[:, 1:2]
        x3 = input[:, 34:]
        b = x1.size(0)

        x3 = x3.reshape(b*2, 2, 48, 48)
        x3_on = torch.zeros_like(x3)

        # ===== motion magnification =====
        x1 = self.man_L(self.enc_L(x1_on), self.enc_L(x1), self.amp_factor)
        x2 = self.man_S(self.enc_S(x2_on), self.enc_S(x2), self.amp_factor)
        x3 = self.man_T(self.enc_T(x3_on), self.enc_T(x3), self.amp_factor)

        # ===== shallow conv =====
        x1 = self.pool(self.conv1_L(x1))
        x2 = self.pool(self.conv1_S(x2))
        x3 = self.pool(self.conv1_T(x3))
        x1 = self.eca(x1)

        # ===== deeper conv =====
        x1 = self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(x1))))))
        x2 = self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(x2))))))
        x3 = self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(x3))))))

        # ===== final conv =====
        x1 = self.global_pool(self.relu(self.bn5(self.conv5(self.relu(self.bn4(self.conv4(x1)))))))
        x2 = self.global_pool(self.relu(self.bn5(self.conv5(self.relu(self.bn4(self.conv4(x2)))))))
        x3 = self.global_pool(self.relu(self.bn5(self.conv5(self.relu(self.bn4(self.conv4(x3)))))))

        x1 = x1.reshape(b, -1)
        x2 = x2.reshape(b, -1)
        x3 = x3.reshape(b, 2, -1)
        x3 = self.temporal_attn(x3)
        x3 = x3.mean(dim=1)

        # ===== cross-branch fusion =====
        final_feature = self.cross_attn(x1, x2, x3)
        final_feature = self.dropout(final_feature)

        # projection for feature distillation
        final_feature_proj = self.proj(final_feature)

        # AC1/AC2
        AC1_feature = x1
        AC2_feature = x2
        AC1_out = self.fc_ac1(AC1_feature)
        AC2_out = self.fc_ac2(AC2_feature)

        # final classification
        yhat = self.fc(final_feature)

        return yhat, AC1_out, AC2_out, final_feature, final_feature_proj, AC1_feature, AC2_feature

    # ===== prototype EMA update =====
    @torch.no_grad()
    def update_prototypes(self, features, labels, momentum=0.9):
        for i in range(self.prototypes.size(0)):
            mask = labels == i
            if mask.sum() == 0:
                continue
            feat_mean = features[mask].mean(dim=0)
            self.proto_ema[i] = momentum * self.proto_ema[i] + (1 - momentum) * feat_mean
            self.prototypes.data[i] = self.proto_ema[i]


# =========================
# get_model
# =========================
def get_model(model_name, class_num, alpha):
    if model_name == "SKD_TSTSAN":
        return SKD_TSTSAN(class_num, alpha)