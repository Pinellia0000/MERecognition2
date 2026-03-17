import torch
import torch.nn as nn
import math

"""
在 all_model_4.py 的基础上继续优化：
1) 删除硬编码尺寸（48x48）；
2) 统一 stage 构造：make_bottleneck_stage / make_temporal_stage；
3) TemporalShift 稳健化：clone() + fold 最小为 1；
4) 共享注意力模块保留，避免冗余实例化；
5) 初始化覆盖 LayerNorm；加载权重用 map_location='cpu'；
6) TemporalAttention 支持任意 n_segment；
7) 接口与返回值顺序保持不变。
"""

# -------------------------
# 工具函数
# -------------------------

def gen_state_dict(weights_path):
    st = torch.load(weights_path, map_location='cpu')
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
    def shift(x, n_segment, fold_div=8, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.reshape(n_batch, n_segment, c, h, w)

        fold = max(1, c // fold_div)
        if inplace:
            # 一般不建议 inplace，保持计算图清晰
            raise NotImplementedError("In-place shift is disabled for stability.")
        else:
            out = x.clone()
            # 左移：t -> t-1
            out[:, :-1, :fold] = x[:, 1:, :fold]
            # 右移：t -> t+1
            out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
            # 中间通道不变
            # out[:, :, 2*fold:] = x[:, :, 2*fold:]  # clone() 已含原值，无需再次赋值

        return out.reshape(nt, c, h, w)


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
    通用时序注意力：支持任意 n_segment
    输入：[B*n_segment, C, H, W] 或 [B*n_segment, C, 1, 1]
    输出：[B, C, H, W]
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
        assert bt % n_segment == 0, f"bt({bt}) must be divisible by n_segment({n_segment})"
        b = bt // n_segment
        x = x.reshape(b, n_segment, c, h, w)
        # 全局池化得到每段的描述 [B, n_segment, C]
        desc = x.mean(dim=[3, 4])
        # 两层 MLP 得到每段的标量权重
        logits = self.fc2(self.act(self.fc1(desc)))  # [B, n_segment, 1]
        weights = self.softmax(logits.squeeze(-1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, n_segment, 1, 1, 1]
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
            nn.LayerNorm(in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(in_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.head(x)


# -------------------------
# Stage 生成器
# -------------------------
def make_bottleneck_stage(in_ch: int, out_ch: int, num_blocks: int,
                          use_eca_first: bool = False, use_sa_first: bool = False,
                          use_eca_others: bool = False, use_sa_others: bool = False):
    """普通（非时序）stage：第一个 block 负责通道变换"""
    blocks = []
    # 第一个 block：in_ch->out_ch
    blocks.append(Bottleneck(in_ch, out_ch, use_eca=use_eca_first, use_sa=use_sa_first))
    # 其余 block：out_ch->out_ch
    for _ in range(num_blocks - 1):
        blocks.append(Bottleneck(out_ch, out_ch, use_eca=use_eca_others, use_sa=use_sa_others))
    return nn.Sequential(*blocks)


def make_temporal_stage(in_ch: int, out_ch: int, num_blocks: int,
                        n_segment: int = 2, n_div: int = 8,
                        use_sa: bool = True):
    """时序 stage：每个 block 外包一层 TemporalShift"""
    blocks = []
    # 第一个 block：in_ch->out_ch
    blocks.append(TemporalShift(Bottleneck(in_ch, out_ch, use_sa=use_sa), n_segment=n_segment, n_div=n_div))
    # 其余 block：out_ch->out_ch
    for _ in range(num_blocks - 1):
        blocks.append(TemporalShift(Bottleneck(out_ch, out_ch, use_sa=use_sa), n_segment=n_segment, n_div=n_div))
    return nn.Sequential(*blocks)


class SKD_TSTSAN_v2(nn.Module):
    """
    兼容原接口：SKD_TSTSAN_v2(num_classes, amp_factor)
    主要改动：
      1) 共享注意力：每个 stage 共享 ECA/SA，减少冗余参数。
      2) 引入残差瓶颈，替换重复 Conv+BN+ReLU，特征更稳定。
      3) 通用 TemporalAttention（任意 n_segment），增强时序聚合。
      4) 去掉尺寸硬编码，forward 对任意输入 HxW 更鲁棒。
      5) 分类头：LN+ReLU+Dropout(0.4)+FC。
    """
    def __init__(self, num_classes: int = 5, amp_factor: int = 5, n_segment: int = 2):
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

        # --------- 使用 stage 生成器构建网络骨干 ----------
        # AC1: 64 -> 128，2 个 block
        self.ac1_L = make_bottleneck_stage(64, 128, num_blocks=2, use_eca_first=True, use_sa_first=True,
                                           use_eca_others=True, use_sa_others=True)
        self.ac1_S = make_bottleneck_stage(64, 128, num_blocks=2, use_sa_first=True, use_sa_others=True)
        self.ac1_T = make_temporal_stage(64, 128, num_blocks=2, n_segment=n_segment, n_div=8, use_sa=True)
        self.ac1_pool = nn.AdaptiveAvgPool2d(1)
        self.ta_128 = TemporalAttention(128)
        self.ac1_head = ClassifierHead(128 * 3, num_classes, p=0.4)

        # 中间层：再堆叠两次 64 通道的瓶颈，最后降采样
        self.mid_L = nn.Sequential(
            make_bottleneck_stage(64, 64, num_blocks=2, use_eca_first=True, use_eca_others=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.mid_S = nn.Sequential(
            make_bottleneck_stage(64, 64, num_blocks=2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.mid_T = nn.Sequential(
            make_temporal_stage(64, 64, num_blocks=2, n_segment=n_segment, n_div=8, use_sa=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

        # AC2：64 -> 128，再分类头
        self.ac2_L = make_bottleneck_stage(64, 128, num_blocks=2, use_eca_first=True, use_eca_others=True)
        self.ac2_S = make_bottleneck_stage(64, 128, num_blocks=2)
        self.ac2_T = make_temporal_stage(64, 128, num_blocks=2, n_segment=n_segment, n_div=8, use_sa=True)
        self.ac2_pool = nn.AdaptiveAvgPool2d(1)
        self.ta_128_b = TemporalAttention(128)
        self.ac2_head = ClassifierHead(128 * 3, num_classes, p=0.4)

        # 最终：再 128×2 瓶颈并汇聚
        self.final_L = make_bottleneck_stage(64, 128, num_blocks=2, use_eca_first=True, use_eca_others=True)
        self.final_S = make_bottleneck_stage(64, 128, num_blocks=2)
        self.final_T = make_temporal_stage(64, 128, num_blocks=2, n_segment=n_segment, n_div=8, use_sa=True)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.ta_128_c = TemporalAttention(128)
        self.final_head = ClassifierHead(128 * 3, num_classes, p=0.4)

        self.consensus = ConsensusModule('avg')  # 备用：与原版保持一致
        self.amp_factor = amp_factor
        self.n_segment = n_segment

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
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        """
        input 形状假设与原版一致：
          - x1: 16 通道（2:18）
          - x1_onset: 16 通道（18:34）
          - x2: 单通道（0）, onset（1）
          - x3: 4 通道（34:），按照 [B*n_segment, 2, H, W] 使用，默认 n_segment=2
        """
        # 切分输入（不依赖具体 H,W）
        x1 = input[:, 2:18, :, :]
        x1_onset = input[:, 18:34, :, :]
        x2 = input[:, 0, :, :].unsqueeze(1)
        x2_onset = input[:, 1, :, :].unsqueeze(1)
        x3 = input[:, 34:, :, :]  # [B, 4, H, W]（默认 2 段、每段 2 通道）

        bsz, ch3, H, W = x3.size()
        assert ch3 % 2 == 0, f"x3 channel({ch3}) must be divisible by 2"
        n_segment = self.n_segment
        # 将 [B, 4, H, W] -> [B*n_segment, 2, H, W]（通用：假设每段 2 通道）
        x3 = x3.reshape(bsz * n_segment, 2, H, W)

        # 设备对齐（修复原版 .cuda() 绑死设备）；onset 用全 0 与 x3 同形状
        dev = input.device
        x3_onset = torch.zeros_like(x3, device=dev)

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
        ac1_x3 = self.ta_128(ac1_x3, n_segment=n_segment)  # 时序注意力
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
        ac2_x3 = self.ta_128_b(ac2_x3, n_segment=n_segment)
        ac2_x1_g = self._global_pool_flat(self.ac2_pool(ac2_x1))
        ac2_x2_g = self._global_pool_flat(self.ac2_pool(ac2_x2))
        ac2_x3_g = self._global_pool_flat(self.ac2_pool(ac2_x3))
        ac2_feat = torch.cat([ac2_x1_g, ac2_x2_g, ac2_x3_g], dim=1)  # [B, 384]
        ac2_logits = self.ac2_head(ac2_feat)

        # ---------------- 最终汇聚 ----------------
        f1 = self.final_L(x1m)
        f2 = self.final_S(x2m)
        f3 = self.final_T(x3m)
        f3 = self.ta_128_c(f3, n_segment=n_segment)

        f1_g = self._global_pool_flat(self.final_pool(f1))
        f2_g = self._global_pool_flat(self.final_pool(f2))
        f3_g = self._global_pool_flat(self.final_pool(f3))
        final_feat = torch.cat([f1_g, f2_g, f3_g], dim=1)  # [B, 384]
        final_logits = self.final_head(final_feat)

        # 兼容原版返回顺序
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
