from os import path
import os
import numpy as np
import cv2
import time
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

# 注意：从你的 model 文件导入 get_model 等（保持原来结构）
from model.all_model_3 import *

all_model_path = "/kaggle/working/MERecognition/model/all_model_3.py"

"""
保持原有输出格式（每个 subject 的打印、Best Predicted、Ground Truth、UF1/UAR、best UF1/best UAR、每个 subject 的 class count 等），
并加入以下改进（不会改变原有指标计算逻辑，只增强训练稳定性与可复现性）：
1)Logger 增加 close()，最后安全关闭日志。
2)支持自动类别权重（若需）。
3)支持 FocalLoss（含可选权重）。
4)iteration 级的 warmup + CosineAnnealingLR。
5)梯度裁剪、训练/验证日志、保存完整 checkpoint（可选）。
6)额外输出：smoothed（多数投票）UF1/UAR（仅用于“额外报告”，不会替换原结果）。
7)保留并打印 all_accuracy_dict 和总耗时。
"""

# ---------- 工具函数 / 小模块 ----------
def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def confusionMatrix(config, gt, pred, show=False):
    """
    与你原来一致：多分类宏平均 F1 和宏平均 Recall（UF1, UAR）
    """
    unique_labels = sorted(set(gt) | set(pred))
    if set(unique_labels).issubset({0, 1}):
        labels = [0, 1]
    else:
        labels = list(range(config.class_num))
    f1 = f1_score(gt, pred, average='macro', labels=labels, zero_division=0)
    recall = recall_score(gt, pred, average='macro', labels=labels, zero_division=0)
    return f1, recall

def normalize_gray(images):
    images = cv2.normalize(images, None, alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return images

def recognition_evaluation(dataset, final_gt, final_pred, show=False):
    """
    直接保留原先 UF1/UAR 计算方式（CASME2 五类映射）
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

# ---------- 损失函数 ----------
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
    elif loss_name == "CELoss_weighted":
        return nn.CrossEntropyLoss(weight=weight)
    else:
        return nn.CrossEntropyLoss(weight=weight)

# ---------- 日志类 ----------
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)  # 行缓冲

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass

# ---------- KD / feature loss ----------
def new_kd_loss_function(output, target_output, temperature):
    """Compute kd loss"""
    output = output / temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = nn.KLDivLoss(reduction="batchmean")(output_log_softmax, target_output)
    return loss_kd

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

# ---------- 其他 ----------
CASME2_numbers = [32, 25, 61, 27, 99]

def majority_vote_smooth(preds, window=3):
    """
    简单预测平滑（用于额外报告，不改变原best/online指标）
    """
    if window <= 1 or len(preds) == 0:
        return preds[:]
    half = window // 2
    out = preds[:]
    for i in range(len(preds)):
        s = max(0, i - half)
        e = min(len(preds), i + half + 1)
        cnt = Counter(preds[s:e])
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
    y_arr = np.array(y_list)
    uniq = np.unique(y_arr)
    missing = [c for c in classes if c not in uniq]
    if len(missing) > 0:
        y_for_weight = np.concatenate([y_arr] + [np.array([m]) for m in missing])
    else:
        y_for_weight = y_arr
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_for_weight)
    return torch.tensor(w, dtype=torch.float, device=device)

def gen_state_dict(weights_path):
    st = torch.load(weights_path, map_location="cpu")
    st_ks = list(st.keys())
    st_vs = list(st.values())
    state_dict = {}
    for st_k, st_v in zip(st_ks, st_vs):
        state_dict[st_k.replace('module.', '')] = st_v
    return state_dict

# ---------- 主训练函数（保留原输出格式） ----------
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

    # config 选项
    auto_class_weight = getattr(config, "auto_class_weight", True)
    focal_gamma = getattr(config, "focal_gamma", 2.0)
    smooth_window = getattr(config, "smooth_window", 3)

    # numbers 初始化（保留原逻辑）
    numbers = []
    dataset_name = os.path.basename(config.main_path).split("_")[0]  # "CASME2"
    if config.loss_function in ["FocalLoss_weighted", "CELoss_weighted"]:
        if dataset_name == "CASME2":
            numbers = CASME2_numbers

    if (config.train):
        exp_root = '/kaggle/working/Experiment_for_recognize/' + config.exp_name
        if not path.exists(exp_root):
            os.makedirs(exp_root)

    current_file = os.path.abspath(__file__)
    try:
        shutil.copy(current_file, '/kaggle/working/Experiment_for_recognize/' + config.exp_name)
    except Exception:
        pass
    try:
        shutil.copy(all_model_path, '/kaggle/working/Experiment_for_recognize/' + config.exp_name)
    except Exception:
        pass

    log_file_path = '/kaggle/working/Experiment_for_recognize/' + config.exp_name + "/log.txt"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    sys.stdout = Logger(log_file_path)

    total_gt = []
    total_pred = []
    best_total_pred = []
    all_accuracy_dict = {}

    t = time.time()

    main_path = config.main_path
    subName = os.listdir(main_path)
    subName.sort()

    # training loop per subject (LOS O)
    for n_subName in subName:
        print('Subject:', n_subName)

        X_train = []
        y_train = []

        X_test = []
        y_test = []

        # read train
        expression = os.listdir(main_path + '/' + n_subName + '/train')
        for n_expression in expression:
            case_list = get_folder_all_cases(main_path + '/' + n_subName + '/train/' + n_expression)
            for case in case_list:
                y_train.append(int(n_expression))

                end_input = []
                large_S = normalize_gray(cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_apex.jpg", 0))
                large_S_onset = normalize_gray(cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_onset.jpg", 0))
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
                            block = large_S[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]
                            scaled_block = cv2.resize(block, (48, 48))
                            end_input.append(scaled_block)

                for grid_size in grid_sizes:
                    height, width = large_S_onset.shape
                    block_height, block_width = height // grid_size, width // grid_size
                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S_onset[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]
                            scaled_block = cv2.resize(block, (48, 48))
                            end_input.append(scaled_block)

                end_input.append(normalize_gray(cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_1_u.jpg", 0)))
                end_input.append(normalize_gray(cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_1_v.jpg", 0)))
                end_input.append(normalize_gray(cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_2_u.jpg", 0)))
                end_input.append(normalize_gray(cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_2_v.jpg", 0)))

                end_input = np.stack(end_input, axis=-1)
                X_train.append(end_input)

        # read test
        expression = os.listdir(main_path + '/' + n_subName + '/test')
        for n_expression in expression:
            case_list = get_folder_all_cases(main_path + '/' + n_subName + '/test/' + n_expression)
            for case in case_list:
                y_test.append(int(n_expression))

                end_input = []
                large_S = normalize_gray(cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_apex.jpg", 0))
                large_S_onset = normalize_gray(cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_onset.jpg", 0))
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
                            block = large_S[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]
                            scaled_block = cv2.resize(block, (48, 48))
                            end_input.append(scaled_block)

                for grid_size in grid_sizes:
                    height, width = large_S_onset.shape
                    block_height, block_width = height // grid_size, width // grid_size
                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S_onset[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]
                            scaled_block = cv2.resize(block, (48, 48))
                            end_input.append(scaled_block)

                end_input.append(normalize_gray(cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_1_u.jpg", 0)))
                end_input.append(normalize_gray(cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_1_v.jpg", 0)))
                end_input.append(normalize_gray(cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_2_u.jpg", 0)))
                end_input.append(normalize_gray(cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_2_v.jpg", 0)))

                end_input = np.stack(end_input, axis=-1)
                X_test.append(end_input)

        # prepare paths & writer
        weight_path = '/kaggle/working/Experiment_for_recognize/' + config.exp_name + '/' + n_subName + '/' + n_subName + '.pth'
        log_path = '/kaggle/working/Experiment_for_recognize/' + config.exp_name + '/' + n_subName + '/' + "logs"
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)

        # model
        model = get_model(config.model, config.class_num, config.Aug_alpha).to(device)

        # 动态 class weight
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

        # 初始化 / 载入预训练
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

        # optimizer & scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.0005)
        # prepare data -> tensors
        X_train = np.array(X_train, dtype=np.float32)
        X_train = torch.from_numpy(X_train).permute(0, 3, 1, 2)
        y_train = torch.tensor(y_train, dtype=torch.long)
        dataset_train = TensorDataset(X_train, y_train)

        def worker_init_fn(worker_id):
            random.seed(seed + worker_id)
            np.random.seed(seed + worker_id)

        train_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)

        X_test = np.array(X_test, dtype=np.float32)
        X_test = torch.from_numpy(X_test).permute(0, 3, 1, 2)
        y_test = torch.tensor(y_test, dtype=torch.long)
        dataset_test = TensorDataset(X_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

        max_iter = config.max_iter
        iter_num = 0
        epochs = max(max_iter // max(1, len(train_dl)) + 1, 1)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.05)
        warmup_epochs = max(1, min(5, epochs // 10))
        warmup_iters = warmup_epochs * max(1, len(train_dl))

        # loss function selection
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

                # iteration-level warmup: lr increases during first warmup_iters iterations
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
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                    # warmup lr by iterations
                    if iter_num < warmup_iters:
                        for pg in optimizer.param_groups:
                            pg['lr'] = learning_rate * (iter_num + 1) / warmup_iters

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

                # scheduler step after warmup iterations finish (optional)
                if iter_num >= warmup_iters:
                    # step with epoch to follow cosine schedule
                    scheduler.step()

            # eval
            model.eval()
            num_val_correct = 0
            middle1_num_val_correct = 0
            middle2_num_val_correct = 0
            num_val_examples = 0
            temp_best_each_subject_pred = []
            temp_y = []

            with torch.no_grad():
                for batch in test_dl:
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    yhat, AC1_out, AC2_out, final_feature, AC1_feature, AC2_feature = model(x)

                    num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    middle1_num_val_correct += (torch.max(AC1_out, 1)[1] == y).sum().item()
                    middle2_num_val_correct += (torch.max(AC2_out, 1)[1] == y).sum().item()

                    num_val_examples += y.shape[0]
                    temp_best_each_subject_pred.extend(torch.max(yhat, 1)[1].tolist())
                    temp_y.extend(y.tolist())

            val_acc = num_val_correct / max(1, num_val_examples)
            middle1_val_acc = middle1_num_val_correct / max(1, num_val_examples)
            middle2_val_acc = middle2_num_val_correct / max(1, num_val_examples)

            writer.add_scalar("Val_Acc", val_acc, epoch)
            writer.add_scalar("Middle1_Val_Acc", middle1_val_acc, epoch)
            writer.add_scalar("Middle2_Val_Acc", middle2_val_acc, epoch)

            if best_accuracy_for_each_subject <= val_acc:
                best_accuracy_for_each_subject = val_acc
                best_each_subject_pred = temp_best_each_subject_pred
                if (config.train) and (config.save_model):
                    # 保存完整 checkpoint（便于恢复）
                    checkpoint = {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch
                    }
                    torch.save(checkpoint, weight_path)

            if not (config.train):
                break

        # print subject results (保持原先输出)
        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {}
        accuracydict['pred'] = best_each_subject_pred
        accuracydict['truth'] = temp_y
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

        # 额外报告：平滑预测的 UF1/UAR（不改变原结果）
        smoothed = majority_vote_smooth(temp_best_each_subject_pred, window=smooth_window)
        total_pred_smoothed = total_pred[:-len(temp_best_each_subject_pred)] + smoothed
        UF1_s, UAR_s = recognition_evaluation(dataset_name, total_gt, total_pred_smoothed, show=False)
        print(f'[Smoothed@{smooth_window}] UF1:', round(UF1_s, 4), '| UAR:', round(UAR_s, 4))

        # per-class counts（如你之前日志）
        cnt_truth = Counter(temp_y)
        cnt_pred = Counter(temp_best_each_subject_pred)
        print('Class count (truth):', dict(cnt_truth))
        print('Class count (pred ) :', dict(cnt_pred))

        writer.close()

    # final evaluation
    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(dataset_name, total_gt, total_pred, show=True)
    best_UF1, best_UAR = recognition_evaluation(dataset_name, total_gt, best_total_pred, show=True)
    print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
    print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    # final smoothed report (只作报告)
    total_pred_smoothed = majority_vote_smooth(total_pred, window=smooth_window)
    UF1_s, UAR_s = recognition_evaluation(dataset_name, total_gt, total_pred_smoothed, show=False)
    print(f'Final [Smoothed@{smooth_window}] UF1:', round(UF1_s, 4), '| UAR:', round(UAR_s, 4))

    elapsed = time.time() - t
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"Total Time Taken: {hours}hours {minutes}minutes {seconds}seconds")
    print(all_accuracy_dict)

    # 关闭日志句柄（安全）
    try:
        if hasattr(sys.stdout, "close"):
            sys.stdout.close()
    except Exception:
        pass

# 如果你希望直接以脚本运行，可以在外部传入 config 对象然后调用 main_SKD_TSTSAN_with_Aug_with_SKD(config)
# 例如：
# if __name__ == "__main__":
#     from your_config_file import config
#     main_SKD_TSTSAN_with_Aug_with_SKD(config)
