import torch
import torch.nn as nn
import math


# =========================
# Encoder
# =========================
class SimpleEncoder(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Motion Gate（抑制噪声）
# =========================
class MotionGate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, onset, motion, amp):
        diff = motion - onset
        gate = self.sigmoid(self.conv(diff))
        return motion + amp * gate * diff


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
class ECA(nn.Module):
    def __init__(self, c):
        super().__init__()
        k = int(abs(math.log2(c) + 1) // 2 * 2 + 1)
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        y = x.mean((2, 3), keepdim=True)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(y)


# =========================
# Temporal Attention（核心）
# =========================
class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, x):
        # x: [B,T,C]
        w = self.fc(x)
        w = torch.softmax(w, dim=1)
        return (x * w).sum(1)


# =========================
# Cross Branch Attention（核心）
# =========================
class CrossBranchAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.q = nn.Linear(c, c)
        self.k = nn.Linear(c, c)
        self.v = nn.Linear(c, c)

    def forward(self, f1, f2, f3):
        F = torch.stack([f1, f2, f3], dim=1)  # [B,3,C]

        Q = self.q(F)
        K = self.k(F)
        V = self.v(F)

        attn = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(F.size(-1)), dim=-1)
        out = attn @ V

        return out[:, 0], out[:, 1], out[:, 2]


# =========================
# 主模型
# =========================
class SKD_TSTSAN(nn.Module):
    def __init__(self, num_classes=5, amp_factor=5, n_segment=2):
        super().__init__()

        self.n_segment = n_segment
        self.amp = amp_factor
        self.amp_factor = amp_factor

        # Encoder
        self.enc_L = SimpleEncoder(16)
        self.enc_S = SimpleEncoder(1)
        self.enc_T = SimpleEncoder(2)

        # Motion Gate
        self.motion_L = MotionGate(32)
        self.motion_S = MotionGate(32)
        self.motion_T = MotionGate(32)

        # Stem（保持5x5更稳定）
        self.conv1_L = nn.Conv2d(32, 64, 5, padding=2)
        self.conv1_S = nn.Conv2d(32, 64, 5, padding=2)
        self.conv1_T = nn.Conv2d(32, 64, 5, padding=2)

        self.bn1_L = nn.BatchNorm2d(64)
        self.bn1_S = nn.BatchNorm2d(64)
        self.bn1_T = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        # TSM
        self.tsm = TemporalShift(nn.Conv2d(64, 64, 3, padding=1), n_segment)

        # deeper
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.eca = ECA(128)

        # Temporal Attention
        self.temp_attn = TemporalAttention(128)

        # Cross Branch
        self.cross = CrossBranchAttention(128)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, input):

        # ===== 数据拆分 =====
        x1 = input[:, 2:18]
        x1_onset = input[:, 18:34]

        x2 = input[:, 0:1]
        x2_onset = input[:, 1:2]

        x3 = input[:, 34:]
        b, c, h, w = x3.shape

        x3 = x3.reshape(b * self.n_segment, 2, h, w)
        x3_onset = torch.zeros_like(x3)

        # ===== Motion =====
        x1 = self.motion_L(self.enc_L(x1_onset), self.enc_L(x1), self.amp)
        x2 = self.motion_S(self.enc_S(x2_onset), self.enc_S(x2), self.amp)
        x3 = self.motion_T(self.enc_T(x3_onset), self.enc_T(x3), self.amp)

        # ===== Stem =====
        x1 = self.pool(self.relu(self.bn1_L(self.conv1_L(x1))))
        x2 = self.pool(self.relu(self.bn1_S(self.conv1_S(x2))))
        x3 = self.pool(self.relu(self.bn1_T(self.conv1_T(x3))))

        # ===== Temporal Shift =====
        x3 = self.tsm(x3)

        # ===== deeper =====
        x1 = self.eca(self.relu(self.bn2(self.conv2(x1))))
        x2 = self.eca(self.relu(self.bn2(self.conv2(x2))))
        x3 = self.eca(self.relu(self.bn2(self.conv2(x3))))

        # ===== GAP =====
        x1_feat = self.gap(x1).flatten(1)
        x2_feat = self.gap(x2).flatten(1)

        # ===== Temporal Attention（关键）=====
        bt, c, h, w = x3.shape
        b = bt // self.n_segment

        x3 = x3.view(b, self.n_segment, c, h, w)
        x3_seq = x3.mean((3, 4))  # [B,T,C]

        x3_feat = self.temp_attn(x3_seq)

        # ===== Cross Branch（关键）=====
        x1_feat, x2_feat, x3_feat = self.cross(x1_feat, x2_feat, x3_feat)

        # ===== 分类 =====
        feat = torch.cat([x1_feat, x2_feat, x3_feat], dim=1)
        feat = self.dropout(feat)
        out = self.fc(feat)

        return out, out, out, feat, feat, feat


# =========================
# 工厂函数
# =========================
def get_model(model_name, class_num, alpha, n_segment=2):
    if model_name == "SKD_TSTSAN":
        return SKD_TSTSAN(class_num, alpha, n_segment)
    raise ValueError(model_name)