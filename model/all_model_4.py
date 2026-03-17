import torch
import torch.nn as nn
import math

"""
在all_model_1.py的基础上

1.共享注意力、减参：把 stage 里重复的 ECA/SA 做了共享，减少冗余模块，推理更稳。

2.时序注意力（TA）：在 T 分支三个阶段（AC1/AC2/Final）都加入了TemporalAttention(2 段)，替代/增强原本的 avg 共识，能自适应给关键帧更高权重。

3.残差瓶颈：把大量 Conv+BN+ReLU 换成Bottleneck 残差块（1×1→3×3→1×1），并在需要处接入 ECA/SA，梯度更顺畅。

4.分类头增强：新增 BN + ReLU + Dropout(0.4) + FC 的 head，和你之前的优化思路对齐。

5.Bug 修复：原代码 x3_onset = torch.zeros(...).cuda() 绑死 GPU，已改为跟随 input.device。

6.初始化：统一做了 Kaiming Normal / BN=1,0 / Linear 正态 初始化，默认更稳。

7.接口保持不变：get_model("SKD_TSTSAN", class_num, alpha) 仍然可用，会返回 v2。
"""

# -------------------------
# 工具函数
# -------------------------

def gen_state_dict(weights_path):
    st = torch.load(weights_path)
    state_dict = {}
    for k, v in st.items():
        state_dict[k.replace('module.', '')] = v
    return state_dict


class ConsensusModule(nn.Module):
    def __init__(self, consensus_type: str, dim: int = 1):
        super().__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, x):
        return SegmentConsensus(self.consensus_type, self.dim)(x)


class SegmentConsensus(nn.Module):
    def __init__(self, consensus_type: str, dim: int = 1):
        super().__init__()
        self.consensus_type = consensus_type
        self.dim = dim

    def forward(self, x):
        if self.consensus_type == 'avg':
            return x.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            return x
        else:
            raise ValueError(f"Unsupported consensus type: {self.consensus_type}")


class TemporalShift(nn.Module):
    def __init__(self, net: nn.Module, n_segment: int = 3, n_div: int = 8, inplace: bool = False):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            raise NotImplementedError
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]          # shift left
            out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]  # shift right
            out[:, :, 2*fold:] = x[:, :, 2*fold:]         # not shift

        return out.view(nt, c, h, w)


class ECALayer2D(nn.Module):
    """ECA 2D：自适应核大小，Avg + Max 两分支"""
    def __init__(self, channel: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs(math.log(channel, 2) + 1) / 2)
        k_size = t if t % 2 else (t + 1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y_avg = self.conv(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_max = self.conv(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y_avg + y_max)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """轻量 CBAM 空间注意力：avg+max -> kxk conv -> sigmoid"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_out))
        return x * attn


class TemporalAttention(nn.Module):
    """
    简易时序注意力（针对 2 段）：
    输入形状 [B*2, C, H, W] 或 [B*2, C, 1, 1]
    输出按注意力对 2 段加权后的 [B, C, H, W]
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.act = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, n_segment: int = 2):
        bt, c, h, w = x.size()
        b = bt // n_segment
        x = x.view(b, n_segment, c, h, w)
        # 全局池化得到每段的描述 [B, 2, C]
        desc = x.mean(dim=[3, 4])
        # 两层 MLP 得到每段的标量权重
        logits = self.fc2(self.act(self.fc1(desc)))  # [B, 2, 1]
        weights = self.softmax(logits.squeeze(-1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 2, 1, 1, 1]
        # 加权求和
        out = (x * weights).sum(dim=1)  # [B, C, H, W]
        return out


class Bottleneck(nn.Module):
    """残差瓶颈：1x1 降维 -> 3x3 -> 1x1 升维，可选 ECA/SA"""
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None,
                 use_eca: bool = False, use_sa: bool = False):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.use_eca = use_eca
        self.use_sa = use_sa
        if use_eca:
            self.eca = ECALayer2D(out_ch)
        if use_sa:
            self.sa = SpatialAttention()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.use_eca:
            out = self.eca(out)
        if self.use_sa:
            out = self.sa(out)
        if self.down is not None:
            identity = self.down(identity)
        out = self.relu(out + identity)
        return out


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, p: float = 0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),  # 替换 BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(in_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.head(x)


class SKD_TSTSAN_v2(nn.Module):
    """
    兼容原接口：SKD_TSTSAN_v2(num_classes, amp_factor)
    主要改动：
      1) 共享注意力：每个 stage 共享 ECA/SA，减少冗余参数。
      2) 引入残差瓶颈，替换重复 Conv+BN+ReLU，特征更稳定。
      3) 引入 TemporalAttention（2 段），替代/增强 avg consensus。
      4) 修复 x3_onset 使用 .cuda() 的设备绑死问题，改为跟随 input.device。
      5) 分类头：BN+ReLU+Dropout(0.4)+FC。
    """
    def __init__(self, num_classes: int = 5, amp_factor: int = 5):
        super().__init__()
        # 延迟导入，避免环境无该包时报错
        from motion_magnification_learning_based_master.magnet import (
            Manipulator as MagManipulator,
            Encoder_No_texture as MagEncoder_No_texture,
        )
        self.Aug_Encoder_L = MagEncoder_No_texture(dim_in=16)
        self.Aug_Encoder_S = MagEncoder_No_texture(dim_in=1)
        self.Aug_Encoder_T = MagEncoder_No_texture(dim_in=2)
        self.Aug_Manipulator_L = MagManipulator()
        self.Aug_Manipulator_S = MagManipulator()
        self.Aug_Manipulator_T = MagManipulator()

        # stem
        self.stem_L = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.stem_S = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.stem_T = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)

        # 共享注意力（stage1 用）
        self.shared_eca_64 = ECALayer2D(64)
        self.shared_sa = SpatialAttention()

        # AC1: 64 -> 128 瓶颈块 ×2 （L/S/T 区分 T 用 TSM）
        self.ac1_L = nn.Sequential(
            Bottleneck(64, 128, use_eca=True, use_sa=True),
            Bottleneck(128, 128, use_eca=True, use_sa=True),
        )
        self.ac1_S = nn.Sequential(
            Bottleneck(64, 128, use_sa=True),
            Bottleneck(128, 128, use_sa=True),
        )
        self.ac1_T = nn.Sequential(
            TemporalShift(Bottleneck(64, 128, use_sa=True), n_segment=2, n_div=8),
            TemporalShift(Bottleneck(128, 128, use_sa=True), n_segment=2, n_div=8),
        )
        self.ac1_pool = nn.AdaptiveAvgPool2d(1)
        self.ta_128 = TemporalAttention(128)  # 用于 T 分支聚合
        self.ac1_head = ClassifierHead(128 * 3, num_classes, p=0.4)

        # 中间层：再堆叠两次 64 通道的瓶颈，最后降采样
        self.mid_L = nn.Sequential(
            Bottleneck(64, 64, use_eca=True),
            Bottleneck(64, 64, use_eca=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.mid_S = nn.Sequential(
            Bottleneck(64, 64),
            Bottleneck(64, 64),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.mid_T = nn.Sequential(
            TemporalShift(Bottleneck(64, 64), n_segment=2, n_div=8),
            TemporalShift(Bottleneck(64, 64), n_segment=2, n_div=8),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

        # AC2：64 -> 128，再分类头
        self.ac2_L = nn.Sequential(
            Bottleneck(64, 128, use_eca=True),
            Bottleneck(128, 128, use_eca=True),
        )
        self.ac2_S = nn.Sequential(
            Bottleneck(64, 128),
            Bottleneck(128, 128),
        )
        self.ac2_T = nn.Sequential(
            TemporalShift(Bottleneck(64, 128), n_segment=2, n_div=8),
            TemporalShift(Bottleneck(128, 128), n_segment=2, n_div=8),
        )
        self.ac2_pool = nn.AdaptiveAvgPool2d(1)
        self.ta_128_b = TemporalAttention(128)
        self.ac2_head = ClassifierHead(128 * 3, num_classes, p=0.4)

        # 最终：再 128×2 瓶颈并汇聚
        self.final_L = nn.Sequential(
            Bottleneck(64, 128, use_eca=True),
            Bottleneck(128, 128, use_eca=True),
        )
        self.final_S = nn.Sequential(
            Bottleneck(64, 128),
            Bottleneck(128, 128),
        )
        self.final_T = nn.Sequential(
            TemporalShift(Bottleneck(64, 128), n_segment=2, n_div=8),
            TemporalShift(Bottleneck(128, 128), n_segment=2, n_div=8),
        )
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.ta_128_c = TemporalAttention(128)
        self.final_head = ClassifierHead(128 * 3, num_classes, p=0.4)

        self.consensus = ConsensusModule('avg')  # 备用：与原版保持一致
        self.amp_factor = amp_factor

        self._init_weights()

    @staticmethod
    def _global_pool_flat(x):
        return x.mean(dim=[2, 3])

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        """
        input 形状假设与原版一致：
          - x1: 16 通道（2:18）
          - x2: 单通道（0）, onset（1）
          - x3: 4 通道（34:），其中实际按照 [B*2, 2, H, W] 使用
        """
        # 切分输入
        x1 = input[:, 2:18, :, :]
        x1_onset = input[:, 18:34, :, :]
        x2 = input[:, 0, :, :].unsqueeze(1)
        x2_onset = input[:, 1, :, :].unsqueeze(1)
        x3 = input[:, 34:, :, :]
        bsz = x1.size(0)
        x3 = x3.reshape(bsz * 2, 2, 48, 48)

        # 设备对齐（修复原版 .cuda() 绑死设备）
        dev = input.device
        x3_onset = torch.zeros(bsz * 2, 2, 48, 48, device=dev)

        # 运动放大编码与操控
        motion_x1_onset = self.Aug_Encoder_L(x1_onset)
        motion_x1 = self.Aug_Encoder_L(x1)
        x1 = self.Aug_Manipulator_L(motion_x1_onset, motion_x1, self.amp_factor)

        motion_x2_onset = self.Aug_Encoder_S(x2_onset)
        motion_x2 = self.Aug_Encoder_S(x2)
        x2 = self.Aug_Manipulator_S(motion_x2_onset, motion_x2, self.amp_factor)

        motion_x3_onset = self.Aug_Encoder_T(x3_onset)
        motion_x3 = self.Aug_Encoder_T(x3)
        x3 = self.Aug_Manipulator_T(motion_x3_onset, motion_x3, self.amp_factor)

        # Stem + 共享注意力
        x1 = self.stem_L(x1); x1 = self.shared_eca_64(x1); x1 = self.shared_sa(x1); x1 = self.pool5(x1)
        x2 = self.stem_S(x2); x2 = self.shared_sa(x2);                x2 = self.pool5(x2)
        x3 = self.stem_T(x3); x3 = self.shared_sa(x3);                x3 = self.pool5(x3)

        # ---------------- AC1 ----------------
        ac1_x1 = self.ac1_L(x1)
        ac1_x2 = self.ac1_S(x2)
        ac1_x3 = self.ac1_T(x3)
        # T 分支时序注意力（2 段）
        ac1_x3 = self.ta_128(ac1_x3, n_segment=2)
        # 池化 + 拼接
        ac1_x1_g = self._global_pool_flat(self.ac1_pool(ac1_x1))
        ac1_x2_g = self._global_pool_flat(self.ac1_pool(ac1_x2))
        ac1_x3_g = self._global_pool_flat(self.ac1_pool(ac1_x3))
        ac1_feat = torch.cat([ac1_x1_g, ac1_x2_g, ac1_x3_g], dim=1)  # [B, 384]
        ac1_logits = self.ac1_head(ac1_feat)

        # ---------------- 中间层（降采样） ----------------
        x1m = self.mid_L(x1)
        x2m = self.mid_S(x2)
        x3m = self.mid_T(x3)

        # ---------------- AC2 ----------------
        ac2_x1 = self.ac2_L(x1m)
        ac2_x2 = self.ac2_S(x2m)
        ac2_x3 = self.ac2_T(x3m)
        ac2_x3 = self.ta_128_b(ac2_x3, n_segment=2)
        ac2_x1_g = self._global_pool_flat(self.ac2_pool(ac2_x1))
        ac2_x2_g = self._global_pool_flat(self.ac2_pool(ac2_x2))
        ac2_x3_g = self._global_pool_flat(self.ac2_pool(ac2_x3))
        ac2_feat = torch.cat([ac2_x1_g, ac2_x2_g, ac2_x3_g], dim=1)  # [B, 384]
        ac2_logits = self.ac2_head(ac2_feat)

        # ---------------- 最终汇聚 ----------------
        f1 = self.final_L(x1m)
        f2 = self.final_S(x2m)
        f3 = self.final_T(x3m)
        f3 = self.ta_128_c(f3, n_segment=2)

        f1_g = self._global_pool_flat(self.final_pool(f1))
        f2_g = self._global_pool_flat(self.final_pool(f2))
        f3_g = self._global_pool_flat(self.final_pool(f3))
        final_feat = torch.cat([f1_g, f2_g, f3_g], dim=1)  # [B, 384]
        final_logits = self.final_head(final_feat)

        # 为了兼容原版返回顺序：
        # return x_all, AC1_x_all, AC2_x_all, final_feature, AC1_feature, AC2_feature
        x_all = final_logits
        AC1_x_all = ac1_logits
        AC2_x_all = ac2_logits
        final_feature = final_feat
        AC1_feature = ac1_feat
        AC2_feature = ac2_feat
        return x_all, AC1_x_all, AC2_x_all, final_feature, AC1_feature, AC2_feature


# -------------------------
# 工厂函数（与原版接口保持一致）
# -------------------------

def get_model(model_name: str, class_num: int, alpha: int):
    if model_name in ["SKD_TSTSAN", "SKD_TSTSAN_v2"]:
        return SKD_TSTSAN_v2(class_num, alpha)
    raise ValueError(f"Unknown model name: {model_name}")
