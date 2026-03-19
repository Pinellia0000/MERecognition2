import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from motion_magnification_learning_based_master.magnet import Manipulator as MagManipulator
from motion_magnification_learning_based_master.magnet import Encoder_No_texture as MagEncoder_No_texture


# =========================
# Cross-Branch Interaction
# =========================
class CrossBranchInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, x3):
        x = torch.stack([x1, x2, x3], dim=1)  # [B, 3, C]

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attn = self.softmax(Q @ K.transpose(-2, -1) / (x.size(-1) ** 0.5))
        out = attn @ V

        return out[:, 0], out[:, 1], out[:, 2]


# =========================
# Temporal Difference
# =========================
class TemporalDiff(nn.Module):
    def forward(self, x):
        # x: [B*2, C, H, W]
        B = x.shape[0] // 2
        x = x.reshape(B, 2, *x.shape[1:])
        diff = x[:, 1] - x[:, 0]
        return diff


# =========================
# ECA
# =========================
class eca_layer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        t = int(abs(math.log(channel, 2) + 1) / 2)
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


# =========================
# 主模型
# =========================
class SKD_TSTSAN(nn.Module):
    def __init__(self, num_classes=5, amp_factor=5):
        super().__init__()

        # motion magnification
        self.enc_L = MagEncoder_No_texture(16)
        self.enc_S = MagEncoder_No_texture(1)
        self.enc_T = MagEncoder_No_texture(2)

        self.manip_L = MagManipulator()
        self.manip_S = MagManipulator()
        self.manip_T = MagManipulator()

        # conv1
        self.conv1_L = nn.Conv2d(32, 64, 5)
        self.conv1_S = nn.Conv2d(32, 64, 5)
        self.conv1_T = nn.Conv2d(32, 64, 5)

        self.bn1_L = nn.BatchNorm2d(64)
        self.bn1_S = nn.BatchNorm2d(64)
        self.bn1_T = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(5, 2, 2)

        # stage2
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # ECA
        self.eca1 = eca_layer(64)
        self.eca2 = eca_layer(64)

        # temporal diff
        self.temporal_diff = TemporalDiff()

        # Cross-branch interaction
        self.cbi = CrossBranchInteraction(64)

        # classifier
        self.fc = nn.Linear(64 * 3, num_classes)

        self.dropout = nn.Dropout(0.2)
        self.amp_factor = amp_factor

    def forward(self, input):

        # ===== 数据拆分 =====
        x1 = input[:, 2:18]
        x1_onset = input[:, 18:34]

        x2 = input[:, 0:1]
        x2_onset = input[:, 1:2]

        x3 = input[:, 34:]
        b = x1.shape[0]

        x3 = x3.reshape(b * 2, 2, 48, 48)
        x3_onset = torch.zeros_like(x3)

        # ===== motion magnification =====
        x1 = self.manip_L(self.enc_L(x1_onset), self.enc_L(x1), self.amp_factor)
        x2 = self.manip_S(self.enc_S(x2_onset), self.enc_S(x2), self.amp_factor)
        x3 = self.manip_T(self.enc_T(x3_onset), self.enc_T(x3), self.amp_factor)

        # ===== temporal difference =====
        x3_diff = self.temporal_diff(x3)
        x3 = x3[:b] + x3_diff

        # ===== conv1 =====
        x1 = self.pool(self.eca1(self.relu(self.bn1_L(self.conv1_L(x1)))))
        x2 = self.pool(self.relu(self.bn1_S(self.conv1_S(x2))))
        x3 = self.pool(self.relu(self.bn1_T(self.conv1_T(x3))))

        # ===== conv2 =====
        x1 = self.eca2(self.relu(self.bn2(self.conv2(x1))))
        x2 = self.relu(self.bn2(self.conv2(x2)))
        x3 = self.relu(self.bn2(self.conv2(x3)))

        x1 = self.relu(self.bn3(self.conv3(x1)))
        x2 = self.relu(self.bn3(self.conv3(x2)))
        x3 = self.relu(self.bn3(self.conv3(x3)))

        # ===== GAP =====
        x1 = self.avgpool(x1).reshape(b, -1)
        x2 = self.avgpool(x2).reshape(b, -1)
        x3 = self.avgpool(x3).reshape(b, -1)

        # ===== Cross-Branch Interaction =====
        x1, x2, x3 = self.cbi(x1, x2, x3)

        # ===== 分类 =====
        feat = torch.cat([x1, x2, x3], dim=1)
        feat = self.dropout(feat)
        out = self.fc(feat)

        return out, feat


# =========================
# 构建接口
# =========================
def get_model(name, num_classes, alpha):
    if name == "SKD_TSTSAN":
        return SKD_TSTSAN(num_classes, alpha)