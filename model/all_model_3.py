import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from motion_magnification_learning_based_master.magnet import Manipulator as MagManipulator
from motion_magnification_learning_based_master.magnet import Encoder_No_texture as MagEncoder_No_texture


"""
改动要点：
1) 共享注意力：shared_ECA_64 / shared_ECA_128 与 shared_SA，减少参数冗余。
2) 残差模块：为 AC1 / AC2 三路都引入可投影的残差单元（支持 TemporalShift 版本）。
3) 多分支融合：最终输出 = (final_out + AC1_out + AC2_out) / 3，更鲁棒。
4) bug 修复：x3_onset 使用 input.device，避免 CPU 报错。
5) 轻量正则：在各分支的分类器前仍保留 dropout。
"""


def gen_state_dict(weights_path):
    st = torch.load(weights_path, map_location="cpu")
    st_ks = list(st.keys())
    st_vs = list(st.values())
    state_dict = {}
    for st_k, st_v in zip(st_ks, st_vs):
        state_dict[st_k.replace('module.', '')] = st_v
    return state_dict


class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)


class SegmentConsensus(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None
        return output


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

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
            out[:, :-1, :fold] = x[:, 1:, :fold]                   # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]              # not shift

        return out.view(nt, c, h, w)


class eca_layer_2d_v2(nn.Module):
    """Efficient Channel Attention for 2D feature maps (avg+max, 1D conv)."""
    def __init__(self, channel):
        super(eca_layer_2d_v2, self).__init__()
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
    """CBAM-like Spatial Attention: avg+max along channel -> 7x7 conv -> sigmoid."""
    def __init__(self, kernel_size=7):
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


class ResidualUnit(nn.Module):
    """
    单层卷积 + BN + ReLU (+ ECA + SA) 的残差单元。
    - 支持 in/out 通道不一致时的投影。
    - 支持把 Conv2d 替换为 TemporalShift(Conv2d)。
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 use_temporal: bool = False,
                 n_segment: int = 2,
                 n_div: int = 8,
                 eca: nn.Module = None,
                 sa: nn.Module = None):
        super().__init__()
        if use_temporal:
            conv_core = TemporalShift(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                n_segment=n_segment, n_div=n_div
            )
        else:
            conv_core = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv = conv_core
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.eca = eca
        self.sa = sa
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        if self.eca is not None:
            out = self.eca(out)
        if self.sa is not None:
            out = self.sa(out)
        if self.proj is not None:
            identity = self.proj(identity)
        return out + identity


class SKD_TSTSAN(nn.Module):
    def __init__(self, out_channels=5, amp_factor=5):
        super(SKD_TSTSAN, self).__init__()
        # === Motion magnification encoders/manipulators ===
        self.Aug_Encoder_L = MagEncoder_No_texture(dim_in=16)
        self.Aug_Encoder_S = MagEncoder_No_texture(dim_in=1)
        self.Aug_Encoder_T = MagEncoder_No_texture(dim_in=2)
        self.Aug_Manipulator_L = MagManipulator()
        self.Aug_Manipulator_S = MagManipulator()
        self.Aug_Manipulator_T = MagManipulator()

        # === Stem: conv1 for each stream ===
        self.conv1_L = nn.Conv2d(32, out_channels=64, kernel_size=5, stride=1, padding=0, bias=False)
        self.conv1_S = nn.Conv2d(32, out_channels=64, kernel_size=5, stride=1, padding=0, bias=False)
        self.conv1_T = nn.Conv2d(32, out_channels=64, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1_L = nn.BatchNorm2d(64)
        self.bn1_S = nn.BatchNorm2d(64)
        self.bn1_T = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)

        # === 共享注意力模块，减少冗余参数 ===
        self.shared_ECA_64 = eca_layer_2d_v2(64)
        self.shared_ECA_128 = eca_layer_2d_v2(128)
        self.shared_SA = SpatialAttention()

        # === AC1：用 ResidualUnit（支持 TemporalShift 的 T 路） ===
        self.AC1_block_L = ResidualUnit(64, 128, use_temporal=False,
                                        eca=self.shared_ECA_128, sa=self.shared_SA)
        self.AC1_block_S = ResidualUnit(64, 128, use_temporal=False,
                                        eca=self.shared_ECA_128, sa=self.shared_SA)
        self.AC1_block_T = ResidualUnit(64, 128, use_temporal=True, n_segment=2, n_div=8,
                                        eca=self.shared_ECA_128, sa=self.shared_SA)
        self.AC1_pool = nn.AdaptiveAvgPool2d(1)
        self.AC1_fc = nn.Linear(in_features=384, out_features=out_channels)

        # === 中间层（保持与原版相近的结构与通道） ===
        self.conv2_L = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_S = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_T = TemporalShift(nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                                     n_segment=2, n_div=8)
        self.bn2_L = nn.BatchNorm2d(64)
        self.bn2_S = nn.BatchNorm2d(64)
        self.bn2_T = nn.BatchNorm2d(64)

        self.conv3_L = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_S = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_T = TemporalShift(nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                                     n_segment=2, n_div=8)
        self.bn3_L = nn.BatchNorm2d(64)
        self.bn3_S = nn.BatchNorm2d(64)
        self.bn3_T = nn.BatchNorm2d(64)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # === AC2：同样用 ResidualUnit ===
        self.AC2_block_L = ResidualUnit(64, 128, use_temporal=False,
                                        eca=self.shared_ECA_128, sa=self.shared_SA)
        self.AC2_block_S = ResidualUnit(64, 128, use_temporal=False,
                                        eca=self.shared_ECA_128, sa=self.shared_SA)
        self.AC2_block_T = ResidualUnit(64, 128, use_temporal=True, n_segment=2, n_div=8,
                                        eca=self.shared_ECA_128, sa=self.shared_SA)
        self.AC2_pool = nn.AdaptiveAvgPool2d(1)
        self.AC2_fc = nn.Linear(in_features=384, out_features=out_channels)

        # === Final head ===
        self.all_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv4_L = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_S = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_T = TemporalShift(nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
                                     n_segment=2, n_div=8)
        self.bn4_L = nn.BatchNorm2d(128)
        self.bn4_S = nn.BatchNorm2d(128)
        self.bn4_T = nn.BatchNorm2d(128)

        self.conv5_L = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_S = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_T = TemporalShift(nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
                                     n_segment=2, n_div=8)
        self.bn5_L = nn.BatchNorm2d(128)
        self.bn5_S = nn.BatchNorm2d(128)
        self.bn5_T = nn.BatchNorm2d(128)

        self.fc2 = nn.Linear(in_features=384, out_features=out_channels)

        # 共识与放大因子
        self.amp_factor = amp_factor
        self.consensus = ConsensusModule("avg")

        # 正则
        self.dropout_feat = nn.Dropout(0.30)
        self.dropout_cls = nn.Dropout(0.20)

    def forward(self, input):
        # ===== 通道切分 =====
        # x1: L 通道（16），x1_onset（16）；x2: S 通道（1），x2_onset（1）；x3: T 通道（后续）
        x1 = input[:, 2:18, :, :]
        x1_onset = input[:, 18:34, :, :]
        x2 = input[:, 0, :, :].unsqueeze(dim=1)
        x2_onset = input[:, 1, :, :].unsqueeze(dim=1)
        x3 = input[:, 34:, :, :]

        bsz = x1.shape[0]
        device = input.device

        # x3 两帧堆叠：形状 [B*2, 2, 48, 48]
        x3 = torch.reshape(x3, (bsz * 2, 2, 48, 48))
        # onset 用 0（使用 input.device 避免 CPU 报错）
        x3_onset = torch.zeros(bsz * 2, 2, 48, 48, device=device)

        # ===== Motion Magnification =====
        motion_x1_onset = self.Aug_Encoder_L(x1_onset)
        motion_x1 = self.Aug_Encoder_L(x1)
        x1 = self.Aug_Manipulator_L(motion_x1_onset, motion_x1, self.amp_factor)

        motion_x2_onset = self.Aug_Encoder_S(x2_onset)
        motion_x2 = self.Aug_Encoder_S(x2)
        x2 = self.Aug_Manipulator_S(motion_x2_onset, motion_x2, self.amp_factor)

        motion_x3_onset = self.Aug_Encoder_T(x3_onset)
        motion_x3 = self.Aug_Encoder_T(x3)
        x3 = self.Aug_Manipulator_T(motion_x3_onset, motion_x3, self.amp_factor)

        # ===== Stem：conv1 + BN + ReLU + (共享 ECA/SA) + MaxPool =====
        x1 = self.conv1_L(x1); x1 = self.bn1_L(x1); x1 = self.relu(x1); x1 = self.shared_ECA_64(x1); x1 = self.shared_SA(x1); x1 = self.maxpool(x1)
        x2 = self.conv1_S(x2); x2 = self.bn1_S(x2); x2 = self.relu(x2); x2 = self.shared_ECA_64(x2); x2 = self.shared_SA(x2); x2 = self.maxpool(x2)
        x3 = self.conv1_T(x3); x3 = self.bn1_T(x3); x3 = self.relu(x3); x3 = self.shared_ECA_64(x3); x3 = self.shared_SA(x3); x3 = self.maxpool(x3)

        # ===== AC1：残差块（输出通道 128）+ GAP -> cat -> FC =====
        AC1_x1 = self.AC1_block_L(x1)
        AC1_x2 = self.AC1_block_S(x2)
        AC1_x3 = self.AC1_block_T(x3)

        AC1_x1 = self.AC1_pool(AC1_x1); AC1_x1_all = AC1_x1.view(AC1_x1.size(0), -1)  # [B, 128]
        AC1_x2 = self.AC1_pool(AC1_x2); AC1_x2_all = AC1_x2.view(AC1_x2.size(0), -1)  # [B, 128]
        AC1_x3 = self.AC1_pool(AC1_x3); AC1_x3_all = AC1_x3.view(AC1_x3.size(0), -1)  # [B*2, 128]

        # 时间共识（T 路 2 段）
        AC1_x3_all = AC1_x3_all.view((-1, 2) + AC1_x3_all.size()[1:])  # [B, 2, 128]
        AC1_x3_all = self.consensus(AC1_x3_all).squeeze(1)            # [B, 128]

        AC1_feature = torch.cat((AC1_x1_all, AC1_x2_all, AC1_x3_all), dim=1)  # [B, 384]
        AC1_x_all = self.dropout_cls(AC1_feature)
        AC1_x_all = self.AC1_fc(AC1_x_all)  # 分支输出1

        # ===== 中间层 =====
        x1 = self.conv2_L(x1); x1 = self.bn2_L(x1); x1 = self.relu(x1)
        x1 = self.conv3_L(x1); x1 = self.bn3_L(x1); x1 = self.relu(x1); x1 = self.shared_ECA_64(x1); x1 = self.shared_SA(x1)
        x1 = self.avgpool(x1)

        x2 = self.conv2_S(x2); x2 = self.bn2_S(x2); x2 = self.relu(x2)
        x2 = self.conv3_S(x2); x2 = self.bn3_S(x2); x2 = self.relu(x2); x2 = self.shared_ECA_64(x2); x2 = self.shared_SA(x2)
        x2 = self.avgpool(x2)

        x3 = self.conv2_T(x3); x3 = self.bn2_T(x3); x3 = self.relu(x3)
        x3 = self.conv3_T(x3); x3 = self.bn3_T(x3); x3 = self.relu(x3); x3 = self.shared_ECA_64(x3); x3 = self.shared_SA(x3)
        x3 = self.avgpool(x3)

        # ===== AC2：残差块（输出通道 128）+ GAP -> cat -> FC =====
        AC2_x1 = self.AC2_block_L(x1)
        AC2_x2 = self.AC2_block_S(x2)
        AC2_x3 = self.AC2_block_T(x3)

        AC2_x1 = self.AC2_pool(AC2_x1); AC2_x1_all = AC2_x1.view(AC2_x1.size(0), -1)
        AC2_x2 = self.AC2_pool(AC2_x2); AC2_x2_all = AC2_x2.view(AC2_x2.size(0), -1)
        AC2_x3 = self.AC2_pool(AC2_x3); AC2_x3_all = AC2_x3.view(AC2_x3.size(0), -1)

        AC2_x3_all = AC2_x3_all.view((-1, 2) + AC2_x3_all.size()[1:])
        AC2_x3_all = self.consensus(AC2_x3_all).squeeze(1)

        AC2_feature = torch.cat((AC2_x1_all, AC2_x2_all, AC2_x3_all), dim=1)  # [B, 384]
        AC2_x_all = self.dropout_cls(AC2_feature)
        AC2_x_all = self.AC2_fc(AC2_x_all)  # 分支输出2

        # ===== Final Head =====
        x1 = self.conv4_L(x1); x1 = self.bn4_L(x1); x1 = self.relu(x1); x1 = self.shared_ECA_128(x1); x1 = self.shared_SA(x1)
        x1 = self.conv5_L(x1); x1 = self.bn5_L(x1); x1 = self.relu(x1); x1 = self.shared_ECA_128(x1)
        x1 = self.all_avgpool(x1); x1_all = x1.view(x1.size(0), -1)  # [B, 128]

        x2 = self.conv4_S(x2); x2 = self.bn4_S(x2); x2 = self.relu(x2); x2 = self.shared_ECA_128(x2); x2 = self.shared_SA(x2)
        x2 = self.conv5_S(x2); x2 = self.bn5_S(x2); x2 = self.relu(x2); x2 = self.shared_ECA_128(x2)
        x2 = self.all_avgpool(x2); x2_all = x2.view(x2.size(0), -1)  # [B, 128]

        x3 = self.conv4_T(x3); x3 = self.bn4_T(x3); x3 = self.relu(x3); x3 = self.shared_ECA_128(x3); x3 = self.shared_SA(x3)
        x3 = self.conv5_T(x3); x3 = self.bn5_T(x3); x3 = self.relu(x3); x3 = self.shared_ECA_128(x3)
        x3 = self.all_avgpool(x3); x3_all = x3.view(x3.size(0), -1)  # [B*2, 128]

        x3_all = x3_all.view((-1, 2) + x3_all.size()[1:])
        x3_all = self.consensus(x3_all).squeeze(1)  # [B, 128]

        final_feature = torch.cat((x1_all, x2_all, x3_all), dim=1)  # [B, 384]
        out_final = self.fc2(self.dropout_feat(final_feature))      # 主分支

        # ===== 多分支融合 =====
        out_all = (out_final + AC1_x_all + AC2_x_all) / 3.0

        # 为保持兼容性，返回顺序与原版一致
        return out_all, AC1_x_all, AC2_x_all, final_feature, AC1_feature, AC2_feature


def get_model(model_name, class_num, alpha):
    if model_name == "SKD_TSTSAN":
        return SKD_TSTSAN(class_num, alpha)
    raise ValueError(f"Unknown model_name: {model_name}")
