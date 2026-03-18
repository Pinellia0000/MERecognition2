import torch
import torch.nn as nn

# =========================
# Motion Magnification
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


class SimpleManipulator(nn.Module):
    def forward(self, onset, motion, amp):
        return motion + amp * (motion - onset)


# =========================
# Temporal Difference
# =========================
class TemporalDifference(nn.Module):
    def forward(self, x, n_segment):
        bt, c, h, w = x.size()
        b = bt // n_segment

        x = x.reshape(b, n_segment, c, h, w)

        diff = x[:, 1:] - x[:, :-1]
        pad = torch.zeros_like(diff[:, :1])
        diff = torch.cat([pad, diff], dim=1)

        return (x + diff).reshape(bt, c, h, w)


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

        x = x.reshape(n_batch, self.n_segment, c, h, w)

        fold = max(1, c // self.fold_div)
        out = x.clone()

        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]

        return self.net(out.reshape(nt, c, h, w))


# =========================
# Cross Branch Fusion
# =========================
class CrossBranchFusion(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.conv = nn.Conv2d(c * 3, c, 1)
        self.bn = nn.BatchNorm2d(c)
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], 1)
        x = self.relu(self.bn(self.conv(x)))
        return x1 + x, x2 + x, x3 + x


# =========================
# Block
# =========================
class Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.down = None
        if in_c != out_c:
            self.down = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        identity = x
        out = self.conv(x)

        if self.down:
            identity = self.down(identity)

        return out + identity


# =========================
# 主模型（去掉 TA）
# =========================
class SKD_TSTSAN_v2_no_TA(nn.Module):
    def __init__(self, num_classes=5, amp_factor=5, n_segment=2):
        super().__init__()

        self.amp = amp_factor
        self.amp_factor = amp_factor
        self.n_segment = n_segment

        self.Aug_Encoder_L = SimpleEncoder(16)
        self.Aug_Encoder_S = SimpleEncoder(1)
        self.Aug_Encoder_T = SimpleEncoder(2)

        self.Aug_Manipulator_L = SimpleManipulator()
        self.Aug_Manipulator_S = SimpleManipulator()
        self.Aug_Manipulator_T = SimpleManipulator()

        self.stem = nn.Conv2d(32, 64, 3, padding=1)

        self.layer1 = Block(64, 64)
        self.layer2 = Block(64, 64)

        self.temporal_shift = TemporalShift(Block(64, 64), n_segment)
        self.temp_diff = TemporalDifference()
        self.cross = CrossBranchFusion(64)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * 3, num_classes)

    def forward(self, input):

        # ===== 分支划分 =====
        x1 = input[:, 2:18]
        x1_onset = input[:, 18:34]

        x2 = input[:, 0:1]
        x2_onset = input[:, 1:2]

        x3 = input[:, 34:]
        b, c, h, w = x3.shape

        assert c == 2 * self.n_segment

        # ===== reshape 成时序 =====
        x3 = x3.reshape(b * self.n_segment, 2, h, w)
        x3_onset = torch.zeros_like(x3)

        # ===== motion =====
        x1 = self.Aug_Manipulator_L(
            self.Aug_Encoder_L(x1_onset),
            self.Aug_Encoder_L(x1), self.amp)

        x2 = self.Aug_Manipulator_S(
            self.Aug_Encoder_S(x2_onset),
            self.Aug_Encoder_S(x2), self.amp)

        x3 = self.Aug_Manipulator_T(
            self.Aug_Encoder_T(x3_onset),
            self.Aug_Encoder_T(x3), self.amp)

        # ===== stem =====
        x1 = self.stem(x1)
        x2 = self.stem(x2)
        x3 = self.stem(x3)

        # ===== 时序 =====
        x3 = self.temp_diff(x3, self.n_segment)

        x1 = self.layer1(x1)
        x2 = self.layer1(x2)
        x3 = self.temporal_shift(x3)

        # ===== reshape 回 batch =====
        bt, c, h, w = x3.shape
        b = bt // self.n_segment
        x3 = x3.reshape(b, self.n_segment, c, h, w).mean(1)

        # ===== cross =====
        x1, x2, x3 = self.cross(x1, x2, x3)

        # ===== deeper =====
        x1 = self.layer2(x1)
        x2 = self.layer2(x2)
        x3 = self.layer2(x3)

        # ===== pooling =====
        x1 = self.pool(x1).flatten(1)
        x2 = self.pool(x2).flatten(1)
        x3 = self.pool(x3).flatten(1)   # ✅ 用这个替代 transformer

        # ===== fusion =====
        feat = torch.cat([x1, x2, x3], dim=1)
        out = self.fc(feat)

        return out, out, out, feat, feat, feat


# =========================
# 工厂函数
# =========================
def get_model(model_name, class_num, alpha, n_segment=2):
    if model_name == "SKD_TSTSAN_no_TA":
        return SKD_TSTSAN_v2_no_TA(class_num, alpha, n_segment)
    elif model_name in ["SKD_TSTSAN", "SKD_TSTSAN_v2"]:
        raise ValueError("请使用带TA版本或指定 no_TA")
    raise ValueError(model_name)