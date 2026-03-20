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
# Cross Attention
# =========================
class CrossBranchAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x1, x2, x3):
        q = self.q(x1)
        k = self.k(x3)
        v = self.v(x2)

        attn = torch.matmul(q, k.T) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        return torch.cat([x1, out, x3], dim=1)


# =========================
# Temporal Attention
# =========================
class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        out, _ = self.attn(x, x, x)
        return out.permute(1, 0, 2)


# =========================
# ⭐ 主模型（带 AC1 / AC2）
# =========================
class SKD_TSTSAN(nn.Module):
    def __init__(self, num_classes=5, amp_factor=5):
        super().__init__()

        self.amp = amp_factor

        # Motion Magnification
        self.enc_L = MagEncoder_No_texture(16)
        self.enc_S = MagEncoder_No_texture(1)
        self.enc_T = MagEncoder_No_texture(2)

        self.man_L = MagManipulator()
        self.man_S = MagManipulator()
        self.man_T = MagManipulator()

        # Conv
        self.conv1_L = nn.Conv2d(32, 64, 5)
        self.conv1_S = nn.Conv2d(32, 64, 5)
        self.conv1_T = nn.Conv2d(32, 64, 5)

        self.bn1_L = nn.BatchNorm2d(64)
        self.bn1_S = nn.BatchNorm2d(64)
        self.bn1_T = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(5, 2, 2)

        self.eca = ECA(64)

        # deeper
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        # final
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # attention
        self.temporal_attn = TemporalAttention(128)
        self.cross_attn = CrossBranchAttention(128)

        # 分类头
        self.fc = nn.Linear(128 * 3, num_classes)

        # ⭐ AC1 / AC2 heads
        self.fc_ac1 = nn.Linear(128, num_classes)
        self.fc_ac2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)

        # ⭐ Prototype
        self.prototypes = nn.Parameter(torch.randn(num_classes, 128 * 3))
        self.proj = nn.Linear(384, 128)

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

        # ===== motion =====
        x1 = self.man_L(self.enc_L(x1_on), self.enc_L(x1), self.amp)
        x2 = self.man_S(self.enc_S(x2_on), self.enc_S(x2), self.amp)
        x3 = self.man_T(self.enc_T(x3_on), self.enc_T(x3), self.amp)

        # ===== shallow =====
        x1 = self.pool(self.relu(self.bn1_L(self.conv1_L(x1))))
        x2 = self.pool(self.relu(self.bn1_S(self.conv1_S(x2))))
        x3 = self.pool(self.relu(self.bn1_T(self.conv1_T(x3))))

        x1 = self.eca(x1)

        # ===== deeper =====
        x1 = self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(x1))))))
        x2 = self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(x2))))))
        x3 = self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(x3))))))

        # ===== final conv =====
        x1 = self.global_pool(self.relu(self.bn5(self.conv5(self.relu(self.bn4(self.conv4(x1)))))))
        x2 = self.global_pool(self.relu(self.bn5(self.conv5(self.relu(self.bn4(self.conv4(x2)))))))
        x3 = self.global_pool(self.relu(self.bn5(self.conv5(self.relu(self.bn4(self.conv4(x3)))))))

        x1 = x1.reshape(b, -1)  # (B,128)
        x2 = x2.reshape(b, -1)

        x3 = x3.reshape(b, 2, -1)
        x3 = self.temporal_attn(x3)
        x3 = x3.mean(dim=1)

        # ⭐ 主融合
        final_feature = self.cross_attn(x1, x2, x3)
        final_feature = self.dropout(final_feature)

        yhat = self.fc(final_feature)
        # ⭐ projection（用于 feature distill）
        final_feature_proj = self.proj(final_feature)

        # ⭐ AC1 / AC2
        AC1_feature = x1
        AC2_feature = x2

        AC1_out = self.fc_ac1(AC1_feature)
        AC2_out = self.fc_ac2(AC2_feature)



        return yhat, AC1_out, AC2_out, final_feature, final_feature_proj, AC1_feature, AC2_feature


# =========================
# get_model
# =========================
def get_model(model_name, class_num, alpha):
    if model_name == "SKD_TSTSAN":
        return SKD_TSTSAN(class_num, alpha)