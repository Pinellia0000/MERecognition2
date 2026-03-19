import torch.nn as nn
import torch
import math
from motion_magnification_learning_based_master.magnet import Manipulator as MagManipulator
from motion_magnification_learning_based_master.magnet import Encoder_No_texture as MagEncoder_No_texture


# =========================
# Temporal Difference（核心改进）
# =========================
class TemporalDifference(nn.Module):
    def forward(self, x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment

        x = x.view(n_batch, n_segment, c, h, w)

        diff = x[:, 1:] - x[:, :-1]
        pad = torch.zeros_like(diff[:, :1])
        diff = torch.cat([pad, diff], dim=1)

        return (x + diff).view(nt, c, h, w)


# =========================
# Temporal Shift
# =========================
class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=2, n_div=8):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)

        fold = max(1, c // self.fold_div)
        out = x.clone()

        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]

        return self.net(out.view(nt, c, h, w))


# =========================
# ECA
# =========================
class eca_layer_2d_v2(nn.Module):
    def __init__(self, channel):
        super().__init__()
        t = int(abs(math.log(channel, 2) + 1) / 2)
        k_size = t if t % 2 else (t + 1)
        self.conv = nn.Conv1d(1, 1, k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean((2, 3), keepdim=True)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


# =========================
# 主模型（优化版）
# =========================
class SKD_TSTSAN(nn.Module):
    def __init__(self, out_channels=5, amp_factor=2):
        super().__init__()

        self.n_segment = 2
        self.amp_factor = 2  # ✅ 降低噪声

        # motion
        self.Aug_Encoder_L = MagEncoder_No_texture(dim_in=16)
        self.Aug_Encoder_S = MagEncoder_No_texture(dim_in=1)
        self.Aug_Encoder_T = MagEncoder_No_texture(dim_in=2)

        self.Aug_Manipulator_L = MagManipulator()
        self.Aug_Manipulator_S = MagManipulator()
        self.Aug_Manipulator_T = MagManipulator()

        # 新增 Temporal Difference
        self.temp_diff = TemporalDifference()

        # ===== stem =====
        self.conv1_L = nn.Conv2d(32, 64, 5, padding=2)
        self.conv1_S = nn.Conv2d(32, 64, 5, padding=2)
        self.conv1_T = nn.Conv2d(32, 64, 5, padding=2)

        self.bn1_L = nn.BatchNorm2d(64)
        self.bn1_S = nn.BatchNorm2d(64)
        self.bn1_T = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(5, stride=2, padding=2)

        # ===== AC1 =====
        self.AC1_conv1_T = TemporalShift(nn.Conv2d(64, 128, 3, padding=1), 2)
        self.AC1_conv2_T = TemporalShift(nn.Conv2d(128, 128, 3, padding=1), 2)

        self.AC1_conv1_L = nn.Conv2d(64, 128, 3, padding=1)
        self.AC1_conv1_S = nn.Conv2d(64, 128, 3, padding=1)
        self.AC1_conv2_L = nn.Conv2d(128, 128, 3, padding=1)
        self.AC1_conv2_S = nn.Conv2d(128, 128, 3, padding=1)

        self.AC1_bn1_L = nn.BatchNorm2d(128)
        self.AC1_bn1_S = nn.BatchNorm2d(128)
        self.AC1_bn1_T = nn.BatchNorm2d(128)

        self.AC1_bn2_L = nn.BatchNorm2d(128)
        self.AC1_bn2_S = nn.BatchNorm2d(128)
        self.AC1_bn2_T = nn.BatchNorm2d(128)

        self.AC1_pool = nn.AdaptiveAvgPool2d(1)
        self.AC1_fc = nn.Linear(384, out_channels)

        # ===== middle =====
        self.conv2_L = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_S = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_T = TemporalShift(nn.Conv2d(64, 64, 3, padding=1), 2)

        self.conv3_L = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_S = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_T = TemporalShift(nn.Conv2d(64, 64, 3, padding=1), 2)

        self.bn2_L = nn.BatchNorm2d(64)
        self.bn2_S = nn.BatchNorm2d(64)
        self.bn2_T = nn.BatchNorm2d(64)

        self.bn3_L = nn.BatchNorm2d(64)
        self.bn3_S = nn.BatchNorm2d(64)
        self.bn3_T = nn.BatchNorm2d(64)

        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)

        # ===== final =====
        self.conv4_L = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4_S = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4_T = TemporalShift(nn.Conv2d(64, 128, 3, padding=1), 2)

        self.conv5_L = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5_S = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5_T = TemporalShift(nn.Conv2d(128, 128, 3, padding=1), 2)

        self.bn4_L = nn.BatchNorm2d(128)
        self.bn4_S = nn.BatchNorm2d(128)
        self.bn4_T = nn.BatchNorm2d(128)

        self.bn5_L = nn.BatchNorm2d(128)
        self.bn5_S = nn.BatchNorm2d(128)
        self.bn5_T = nn.BatchNorm2d(128)

        self.fc2 = nn.Linear(384, out_channels)

        self.all_avgpool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, input):

        x1 = input[:, 2:18]
        x2 = input[:, 0:1]
        x3 = input[:, 34:]

        bsz = x1.shape[0]

        x3 = x3.view(bsz * 2, 2, 48, 48)
        x3_onset = torch.zeros_like(x3)  # ✅ 修复

        # motion
        x1 = self.Aug_Manipulator_L(self.Aug_Encoder_L(x1), self.Aug_Encoder_L(x1), self.amp_factor)
        x2 = self.Aug_Manipulator_S(self.Aug_Encoder_S(x2), self.Aug_Encoder_S(x2), self.amp_factor)
        x3 = self.Aug_Manipulator_T(self.Aug_Encoder_T(x3_onset), self.Aug_Encoder_T(x3), self.amp_factor)

        # ===== stem =====
        x1 = self.maxpool(self.relu(self.bn1_L(self.conv1_L(x1))))
        x2 = self.maxpool(self.relu(self.bn1_S(self.conv1_S(x2))))
        x3 = self.maxpool(self.relu(self.bn1_T(self.conv1_T(x3))))

        # ===== Temporal Difference（关键）=====
        x3 = self.temp_diff(x3, 2)

        # ===== AC1 =====
        AC1_x3 = self.AC1_conv1_T(x3)
        AC1_x3 = self.relu(self.AC1_bn1_T(AC1_x3))
        AC1_x3 = self.AC1_conv2_T(AC1_x3)
        AC1_x3 = self.relu(self.AC1_bn2_T(AC1_x3))
        AC1_x3 = self.AC1_pool(AC1_x3).view(bsz, 2, -1).mean(1)

        # ===== middle =====
        x3 = self.temp_diff(x3, 2)
        x3 = self.conv2_T(x3)
        x3 = self.relu(self.bn2_T(x3))

        x3 = self.conv3_T(x3)
        x3 = self.relu(self.bn3_T(x3))
        x3 = self.avgpool(x3)

        # ===== final =====
        x3 = self.temp_diff(x3, 2)
        x3 = self.conv4_T(x3)
        x3 = self.relu(self.bn4_T(x3))

        x3 = self.conv5_T(x3)
        x3 = self.relu(self.bn5_T(x3))

        x3 = self.all_avgpool(x3).view(bsz, 2, -1).mean(1)

        # 简化（重点放T分支）
        out = self.fc2(torch.cat([x3, x3, x3], dim=1))

        return out, out, out, x3, x3, x3


# =========================
# 工厂函数
# =========================
def get_model(model_name, class_num, alpha):
    if model_name == "SKD_TSTSAN":
        return SKD_TSTSAN(class_num, alpha)
    raise ValueError(model_name)