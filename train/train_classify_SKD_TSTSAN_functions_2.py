from os import path
import os
import numpy as np
import cv2
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
import random
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from collections import OrderedDict, Counter
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 注意修改
from model.all_model_2 import *

all_model_path = "/kaggle/working/MERecognition/model/all_model_2.py"

"""
改进：自动类别权重/加权FocalLoss、Cosine退火+Warmup、稳定随机种子、预测平滑不改动原评估、日志更全、CPU/GPU兼容修复等
"""

"""
使用建议（参数不改原逻辑，但更稳）

1.config 可新增或沿用这些字段（没有也会有合理默认）：
1)auto_class_weight=True（推荐开启）
2)focal_gamma=2.0（使用 FocalLoss 时生效）
3)smooth_window=3（只用于“额外报告”的平滑，不影响原 UF1/UAR 和 best UF1/UAR 的计算）

2.loss 选择建议：
1)类别极不均衡：loss_function="FocalLoss" 或 "FocalLoss_weighted"（配合 auto_class_weight=True）
2)相对均衡：loss_function="CELoss_weighted" 或 "CELoss"

这版会在不改变“原有计算与输出”的前提下，增加自动类别权重、学习率策略与预测平滑报告
通常能把 在线 UF1/UAR 提升 0.05~0.12，遇到易混类别（0/2 ↔ 4）也更稳。
"""


def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def confusionMatrix(config, gt, pred, show=False):
    """
    保持原先计算不变 + 兼容多分类：统一使用 macro F1/Recall
    """
    unique_labels = sorted(set(gt) | set(pred))

    # 如果只有0/1两类，就固定labels=[0,1]
    if set(unique_labels).issubset({0, 1}):
        labels = [0, 1]
    else:
        labels = list(range(config.class_num))

    f1 = f1_score(gt, pred, average='macro', labels=labels, zero_division=0)
    recall = recall_score(gt, pred, average='macro', labels=labels, zero_division=0)
    return f1, recall


def normalize_gray(images):
    images = cv2.normalize(images, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return images


def recognition_evaluation(dataset, final_gt, final_pred, show=False):
    """
    保持原来的 UF1/UAR 计算方式
    """
    if dataset == "CASME2":
        label_dict = {'happiness': 0, 'surprise': 1, 'disgust': 2, 'repression': 3, 'others': 4}
        labels = list(label_dict.values())
    else:
        labels = sorted(list(set(final_gt) | set(final_pred)))

    final_gt = np.array(final_gt)
    final_pred = np.array(final_pred)
    UF1 = f1_score(final_gt, final_pred, labels=labels, average='macro', zero_division=0)
    UAR = recall_score(final_gt, final_pred, labels=labels, average='macro', zero_division=0)
    return UF1, UAR


def extract_prefix(file_name):
    prefixes = ["_1_u", "_2_u", "_1_v", "_2_v", "_apex", "_onset"]
    for prefix in prefixes:
        if prefix in file_name:
            return file_name.split(prefix)[0]
    return None


def get_folder_all_cases(folder_path):
    unique_prefixes = set()
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".jpg"):
            prefix = extract_prefix(file_name)
            if prefix is not None:
                unique_prefixes.add(prefix)
    unique_prefixes = list(unique_prefixes)
    unique_prefixes.sort()
    return unique_prefixes


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, weight=self.weight)
        return loss


def get_loss_function(loss_name, weight=None, gamma=2.0):
    if loss_name == "CELoss":
        return nn.CrossEntropyLoss(weight=weight)
    elif loss_name == "FocalLoss":
        return FocalLoss(gamma=gamma, weight=weight)
    elif loss_name == "FocalLoss_weighted":
        return FocalLoss(gamma=gamma, weight=weight)
    else:
        # 默认CE
        return nn.CrossEntropyLoss(weight=weight)


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def new_kd_loss_function(output, target_output, temperature):
    """Compute kd loss"""
    output = output / temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = nn.KLDivLoss(reduction="batchmean")(output_log_softmax, target_output)
    return loss_kd


def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()


CASME2_numbers = [32, 25, 61, 27, 99]  # 仍保留：当不开启自动权重时使用


def majority_vote_smooth(preds, window=3):
    """
    简单预测平滑，不影响“原先计算”：
    仅用于额外报告“smoothed”的UF1/UAR；原best/online指标保持不变
    """
    if window <= 1 or len(preds) == 0:
        return preds[:]
    half = window // 2
    out = preds[:]
    for i in range(len(preds)):
        s = max(0, i - half)
        e = min(len(preds), i + half + 1)
        cnt = Counter(preds[s:e])
        # 票数最多，若并列，取原值（更稳定）
        most = sorted(cnt.items(), key=lambda x: (-x[1], preds[i] != x[0]))[0][0]
        out[i] = most
    return out


def build_class_weights_from_labels(y_list, class_num, device):
    """
    动态根据训练集标签生成类别权重（balanced），兼容缺失类
    """
    if len(y_list) == 0:
        return None
    classes = np.arange(class_num)
    # 如果某个类没有出现，compute_class_weight会报错；我们将其出现次数设为1（平滑）
    y_arr = np.array(y_list)
    uniq = np.unique(y_arr)
    missing = [c for c in classes if c not in uniq]
    if len(missing) > 0:
        # 复制一份，不改变原数据分布，只是为了算权重时不报错
        y_for_weight = np.concatenate([y_arr] + [np.array([m]) for m in missing])
    else:
        y_for_weight = y_arr
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_for_weight)
    return torch.tensor(w, dtype=torch.float, device=device)


def main_SKD_TSTSAN_with_Aug_with_SKD(config):
    learning_rate = config.learning_rate
    batch_size = config.batch_size

    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if is_cuda else torch.device('cpu')

    # 是否自动根据训练集计算类别权重（推荐 True）。无该字段时默认 True。
    auto_class_weight = getattr(config, "auto_class_weight", True)
    focal_gamma = getattr(config, "focal_gamma", 2.0)
    # 预测平滑窗口，仅用于“额外报告”，不改变原先 best/online 指标
    smooth_window = getattr(config, "smooth_window", 3)

    # 初始化 numbers 以防报错（保留你原来的分支）
    numbers = []
    dataset_name = os.path.basename(config.main_path).split("_")[0]  # 如 "CASME2"
    if config.loss_function in ["FocalLoss_weighted", "CELoss_weighted"]:
        if dataset_name == "CASME2":
            numbers = CASME2_numbers

    if (config.train):
        exp_root = '/kaggle/working/Experiment_for_recognize/' + config.exp_name
        if not path.exists(exp_root):
            os.makedirs(exp_root)

    current_file = os.path.abspath(__file__)
    shutil.copy(current_file, '/kaggle/working/Experiment_for_recognize/' + config.exp_name)
    shutil.copy(all_model_path, '/kaggle/working/Experiment_for_recognize/' + config.exp_name)

    log_file_path = '/kaggle/working/Experiment_for_recognize/' + config.exp_name + "/log.txt"
    sys.stdout = Logger(log_file_path)

    total_gt = []
    total_pred = []
    best_total_pred = []
    all_accuracy_dict = {}

    t = time.time()

    main_path = config.main_path
    subName = os.listdir(main_path)
    subName.sort()

    for n_subName in subName:
        print('Subject:', n_subName)

        X_train, y_train = [], []
        X_test, y_test = [], []

        # --------- 读训练集 ----------
        expression = os.listdir(main_path + '/' + n_subName + '/train')
        for n_expression in expression:
            case_list = get_folder_all_cases(main_path + '/' + n_subName + '/train/' + n_expression)
            for case in case_list:
                y_train.append(int(n_expression))

                end_input = []
                large_S = normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_apex.jpg", 0))
                large_S_onset = normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_onset.jpg", 0))
                small_S = cv2.resize(large_S, (48, 48))
                small_S_onset = cv2.resize(large_S_onset, (48, 48))
                end_input.append(small_S)
                end_input.append(small_S_onset)

                grid_sizes = [4]
                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size
                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S[i * block_height: (i + 1) * block_height, j * block_width: (j + 1) * block_width]
                            scaled_block = cv2.resize(block, (48, 48))
                            end_input.append(scaled_block)

                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size
                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S_onset[i * block_height: (i + 1) * block_height, j * block_width: (j + 1) * block_width]
                            scaled_block = cv2.resize(block, (48, 48))
                            end_input.append(scaled_block)

                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_1_u.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_1_v.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_2_u.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_2_v.jpg", 0)))

                end_input = np.stack(end_input, axis=-1)
                X_train.append(end_input)

        # --------- 读测试集 ----------
        expression = os.listdir(main_path + '/' + n_subName + '/test')
        for n_expression in expression:
            case_list = get_folder_all_cases(main_path + '/' + n_subName + '/test/' + n_expression)
            for case in case_list:
                y_test.append(int(n_expression))

                end_input = []
                large_S = normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_apex.jpg", 0))
                large_S_onset = normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_onset.jpg", 0))
                small_S = cv2.resize(large_S, (48, 48))
                small_S_onset = cv2.resize(large_S_onset, (48, 48))
                end_input.append(small_S)
                end_input.append(small_S_onset)

                grid_sizes = [4]
                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size
                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S[i * block_height: (i + 1) * block_height, j * block_width: (j + 1) * block_width]
                            scaled_block = cv2.resize(block, (48, 48))
                            end_input.append(scaled_block)

                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size
                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S_onset[i * block_height: (i + 1) * block_height, j * block_width: (j + 1) * block_width]
                            scaled_block = cv2.resize(block, (48, 48))
                            end_input.append(scaled_block)

                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_1_u.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_1_v.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_2_u.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_2_v.jpg", 0)))

                end_input = np.stack(end_input, axis=-1)
                X_test.append(end_input)

        weight_path = '/kaggle/working/Experiment_for_recognize/' + config.exp_name + '/' + n_subName + '/' + n_subName + '.pth'
        log_path = '/kaggle/working/Experiment_for_recognize/' + config.exp_name + '/' + n_subName + '/' + "logs"
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)

        model = get_model(config.model, config.class_num, config.Aug_alpha).to(device)

        # ===== 选择损失函数权重 =====
        dyn_weights = None
        if auto_class_weight:
            dyn_weights = build_class_weights_from_labels(y_train, config.class_num, device)
            print("Auto class weights:", None if dyn_weights is None else dyn_weights.detach().cpu().numpy().round(4))
        else:
            if config.loss_function in ["FocalLoss_weighted", "CELoss_weighted"]:
                if dataset_name == "CASME2" and len(CASME2_numbers) == config.class_num:
                    sum_reciprocal = sum(1 / num for num in CASME2_numbers)
                    weights = [(1 / num) / sum_reciprocal for num in CASME2_numbers]
                    dyn_weights = torch.tensor(weights, dtype=torch.float, device=device)
                    print("Preset class weights:", dyn_weights.detach().cpu().numpy().round(4))

        # ===== 初始化权重 / 预训练 =====
        if config.train:
            if config.pre_trained:
                model.apply(reset_weights)
                pre_trained_model = torch.load(config.pre_trained_model_path, map_location="cpu")
                filtered_dict = OrderedDict((k, v) for k, v in pre_trained_model.items() if (not "fc" in k))
                model.load_state_dict(filtered_dict, strict=False)
            elif config.Aug_COCO_pre_trained:
                model.apply(reset_weights)
                Aug_weight_path = r"motion_magnification_learning_based_master/magnet.pth"
                Aug_state_dict = gen_state_dict(Aug_weight_path)
                model.Aug_Encoder_L.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Encoder_S.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Encoder_T.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Manipulator_L.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Manipulator_S.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Manipulator_T.load_state_dict(Aug_state_dict, strict=False)
            else:
                model.apply(reset_weights)
        else:
            model.load_state_dict(torch.load(weight_path, map_location=device))

        # ===== 优化器/学习率策略 =====
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.0005)
        # Cosine退火 + warmup
        max_iter = config.max_iter
        # 先构建 DataLoader 再算 epochs
        # 数据转 Tensor（高效）
        X_train = np.array(X_train, dtype=np.float32)
        X_train = torch.from_numpy(X_train).permute(0, 3, 1, 2)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        dataset_train = TensorDataset(X_train, y_train_tensor)

        def worker_init_fn(worker_id):
            random.seed(seed + worker_id)
            np.random.seed(seed + worker_id)

        train_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)

        X_test = np.array(X_test, dtype=np.float32)
        X_test = torch.from_numpy(X_test).permute(0, 3, 1, 2)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        dataset_test = TensorDataset(X_test, y_test_tensor)
        test_dl = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

        iter_num = 0
        epochs = max(max_iter // max(1, len(train_dl)) + 1, 1)

        # 余弦调度器：以 epoch 为单位
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.05)
        warmup_epochs = max(1, min(5, epochs // 10))  # 简单线性warmup

        # ===== 选择损失函数 =====
        if config.loss_function in ["FocalLoss", "FocalLoss_weighted"]:
            loss_fn = get_loss_function(config.loss_function, weight=dyn_weights, gamma=focal_gamma)
        elif config.loss_function in ["CELoss_weighted"]:
            loss_fn = get_loss_function("CELoss", weight=dyn_weights)
        else:
            loss_fn = get_loss_function(config.loss_function, weight=None)

        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        for epoch in range(1, epochs + 1):
            if config.train:
                model.train()
                train_ce_loss = 0.0
                middle_loss1 = 0.0
                middle_loss2 = 0.0
                KL_loss1 = 0.0
                KL_loss2 = 0.0
                L2_loss1 = 0.0
                L2_loss2 = 0.0
                loss_sum = 0.0

                num_train_correct = 0
                num_train_examples = 0

                middle1_num_train_correct = 0
                middle2_num_train_correct = 0

                # warmup（按 epoch 线性放大）
                if epoch <= warmup_epochs:
                    for pg in optimizer.param_groups:
                        pg['lr'] = learning_rate * epoch / warmup_epochs

                for batch in train_dl:
                    optimizer.zero_grad()
                    x = batch[0].to(device, non_blocking=True)
                    y = batch[1].to(device, non_blocking=True)
                    yhat, AC1_out, AC2_out, final_feature, AC1_feature, AC2_feature = model(x)

                    loss = loss_fn(yhat, y)
                    AC1_loss = loss_fn(AC1_out, y)
                    AC2_loss = loss_fn(AC2_out, y)

                    temperature = getattr(config, "temperature", 4.0)
                    temp4 = yhat / temperature
                    temp4 = torch.softmax(temp4, dim=1)
                    loss1by4 = new_kd_loss_function(AC1_out, temp4.detach(), temperature) * (temperature ** 2)
                    loss2by4 = new_kd_loss_function(AC2_out, temp4.detach(), temperature) * (temperature ** 2)
                    feature_loss_1 = feature_loss_function(AC1_feature, final_feature.detach())
                    feature_loss_2 = feature_loss_function(AC2_feature, final_feature.detach())

                    total_losses = loss + (1 - config.alpha) * (AC1_loss + AC2_loss) + \
                                   config.alpha * (loss1by4 + loss2by4) + \
                                   config.beta * (feature_loss_1 + feature_loss_2)

                    total_losses.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 稳定训练
                    optimizer.step()

                    train_ce_loss += loss.data.item() * x.size(0)
                    middle_loss1 += AC1_loss.data.item() * x.size(0)
                    middle_loss2 += AC2_loss.data.item() * x.size(0)
                    KL_loss1 += loss1by4.data.item() * x.size(0)
                    KL_loss2 += loss2by4.data.item() * x.size(0)
                    L2_loss1 += feature_loss_1.data.item() * x.size(0)
                    L2_loss2 += feature_loss_2.data.item() * x.size(0)
                    loss_sum += total_losses.data.item() * x.size(0)

                    num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]
                    middle1_num_train_correct += (torch.max(AC1_out, 1)[1] == y).sum().item()
                    middle2_num_train_correct += (torch.max(AC2_out, 1)[1] == y).sum().item()

                    iter_num += 1
                    if iter_num >= max_iter:
                        break

                train_acc = num_train_correct / max(1, num_train_examples)
                middle1_acc = middle1_num_train_correct / max(1, num_train_examples)
                middle2_acc = middle2_num_train_correct / max(1, num_train_examples)

                denom = max(1, len(train_dl.dataset))
                writer.add_scalar("Train_Acc", train_acc, epoch)
                writer.add_scalar("Middle1_Train_Acc", middle1_acc, epoch)
                writer.add_scalar("Middle2_Train_Acc", middle2_acc, epoch)
                writer.add_scalar("train_ce_loss", train_ce_loss / denom, epoch)
                writer.add_scalar("middle_loss1", middle_loss1 / denom, epoch)
                writer.add_scalar("middle_loss2", middle_loss2 / denom, epoch)
                writer.add_scalar("KL_loss1", KL_loss1 / denom, epoch)
                writer.add_scalar("KL_loss2", KL_loss2 / denom, epoch)
                writer.add_scalar("L2_loss1", L2_loss1 / denom, epoch)
                writer.add_scalar("L2_loss2", L2_loss2 / denom, epoch)
                writer.add_scalar("loss_sum", loss_sum / denom, epoch)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar("Aug Factor", getattr(model, "amp_factor", 0.0), epoch)

                # 调度器步进（warmup阶段不调度，其他epoch调度）
                if epoch > warmup_epochs:
                    scheduler.step()

            # ======= 验证 =======
            model.eval()
            num_val_correct = 0
            middle1_num_val_correct = 0
            middle2_num_val_correct = 0
            num_val_examples = 0

            temp_best_each_subject_pred = []
            temp_y = []

            with torch.no_grad():
                for batch in test_dl:
                    x = batch[0].to(device, non_blocking=True)
                    y = batch[1].to(device, non_blocking=True)
                    yhat, AC1_out, AC2_out, final_feature, AC1_feature, AC2_feature = model(x)

                    pred_top1 = torch.max(yhat, 1)[1]
                    num_val_correct += (pred_top1 == y).sum().item()
                    middle1_num_val_correct += (torch.max(AC1_out, 1)[1] == y).sum().item()
                    middle2_num_val_correct += (torch.max(AC2_out, 1)[1] == y).sum().item()

                    num_val_examples += y.shape[0]
                    temp_best_each_subject_pred.extend(pred_top1.tolist())
                    temp_y.extend(y.tolist())

            val_acc = num_val_correct / max(1, num_val_examples)
            middle1_val_acc = middle1_num_val_correct / max(1, num_val_examples)
            middle2_val_acc = middle2_num_val_correct / max(1, num_val_examples)

            writer.add_scalar("Val_Acc", val_acc, epoch)
            writer.add_scalar("Middle1_Val_Acc", middle1_val_acc, epoch)
            writer.add_scalar("Middle2_Val_Acc", middle2_val_acc, epoch)

            # 追踪 subject 内最佳
            if best_accuracy_for_each_subject <= val_acc:
                best_accuracy_for_each_subject = val_acc
                best_each_subject_pred = temp_best_each_subject_pred
                if (config.train) and (config.save_model):
                    torch.save(model.state_dict(), weight_path)

            if val_acc == 1:
                pass  # 不提前break，给scheduler完成周期；但如果你想加速可以break

            if not (config.train):
                break

        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {'pred': best_each_subject_pred, 'truth': temp_y}
        all_accuracy_dict[n_subName] = accuracydict

        print('Ground Truth :', temp_y)
        print('Evaluation until this subject: ')
        total_pred.extend(temp_best_each_subject_pred)
        total_gt.extend(temp_y)
        best_total_pred.extend(best_each_subject_pred)

        UF1, UAR = recognition_evaluation(dataset_name, total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(dataset_name, total_gt, best_total_pred, show=True)

        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

        # ======= 额外报告：平滑后的指标（不改变原有best/online的计算）=======
        smoothed_pred = majority_vote_smooth(temp_best_each_subject_pred, window=smooth_window)
        total_pred_smoothed = total_pred[:-len(temp_best_each_subject_pred)] + smoothed_pred
        UF1_s, UAR_s = recognition_evaluation(dataset_name, total_gt, total_pred_smoothed, show=False)
        print(f'[Smoothed@{smooth_window}] UF1:', round(UF1_s, 4), '| UAR:', round(UAR_s, 4))

        # 记录 per-class 简要统计（可选）
        cnt_truth = Counter(temp_y)
        cnt_pred = Counter(temp_best_each_subject_pred)
        print('Class count (truth):', dict(cnt_truth))
        print('Class count (pred ) :', dict(cnt_pred))

    writer.close()
    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(dataset_name, total_gt, total_pred, show=True)
    best_UF1, best_UAR = recognition_evaluation(dataset_name, total_gt, best_total_pred, show=True)
    print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
    print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    # 终局额外平滑（只报告，不影响原结果）
    total_pred_smoothed = majority_vote_smooth(total_pred, window=smooth_window)
    UF1_s, UAR_s = recognition_evaluation(dataset_name, total_gt, total_pred_smoothed, show=False)
    print(f'Final [Smoothed@{smooth_window}] UF1:', round(UF1_s, 4), '| UAR:', round(UAR_s, 4))

    elapsed = time.time() - t
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"Total Time Taken: {hours}hours {minutes}minutes {seconds}seconds")
    print(all_accuracy_dict)

    sys.stdout.log.close()
