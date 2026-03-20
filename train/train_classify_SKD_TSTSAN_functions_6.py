from os import path
import os
import numpy as np
import cv2
import time
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import TensorDataset, DataLoader
import random
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import shutil
import sys
import ast
from tqdm import tqdm
# 注意修改
from model.all_model_11 import *

all_model_path = "/kaggle/working/MERecognition2/model/all_model_11.py"
"""
在fuctions_4的基础上
给在for n_subName in subName:和for epoch in range(1, epochs + 1): 加上进度条
"""
# print("DEBUG: ENTER main_SKD_TSTSAN_with_Aug_with_SKD")
# print("__file__ =", __file__)


# =========================
# Prototype Loss
# =========================
def prototype_loss(features, labels, prototypes, T=0.07):
    feat = F.normalize(features, dim=1)
    proto = F.normalize(prototypes, dim=1)

    logits = torch.matmul(feat, proto.T) / T
    return F.cross_entropy(logits, labels)


def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def confusionMatrix(config, gt, pred, show=False):
    """
    gt：真实标签
    pred：预测标签
    按下面这种方法写会有警告

    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)  # 二分类F1的定义
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples  # 正类的召回率

    return f1_score, average_recall

    产生下面这种警告

    /opt/conda/envs/newCondaEnvironment/lib/python3.10/site-packages/sklearn/metrics/_classification.py:534:
    UserWarning: A single label was found in 'y_true' and 'y_pred'.
    For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.
  warnings.warn(

  上面的写法只有在二分类时成立 即分别微表情与非微表情

    """

    unique_labels = sorted(set(gt) | set(pred))

    # 如果只有0/1两类，就固定labels=[0,1]
    if set(unique_labels).issubset({0, 1}):
        labels = [0, 1]
    else:
        # # 获取分类的类别数
        labels = list(range(config.class_num))

    # average='macro' 是宏平均 因为不止二分类 是多情绪分类
    f1 = f1_score(gt, pred, average='macro',
                  labels=labels, zero_division=0)
    recall = recall_score(gt, pred, average='macro',
                          labels=labels, zero_division=0)

    return f1, recall


def normalize_gray(images):
    images = cv2.normalize(images, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return images


def recognition_evaluation(dataset, final_gt, final_pred, show=False):
    """
    这样写效率可能有点低
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''
    """
    if dataset == "CASME2_retinaface_loso_5":
        label_dict = {'happiness': 0, 'surprise': 1, 'disgust': 2, 'repression': 3, 'others': 4}
        labels = list(label_dict.values())
    elif dataset == "CASME2_retinaface_loso_3":
        label_dict = {'positive': 0, 'negative': 1, 'surprise': 2}
        labels = list(label_dict.values())
    elif dataset == "SAMM_retinaface_loso_3":
        label_dict = {'positive': 0, 'negative': 1, 'surprise': 2}
        labels = list(label_dict.values())
    elif dataset == "CASME3_retinaface_loso_7":
        label_dict = {'happy': 0, 'surprise': 1, 'disgust': 2, 'anger': 3, 'fear': 4, 'sad': 5, 'others': 6}
        labels = list(label_dict.values())
    elif dataset == "CASME3_retinaface_loso_4":
        label_dict = {'positive': 0, 'negative': 1, 'surprise': 2, 'others': 3}
        labels = list(label_dict.values())
    elif dataset == "CASME3_retinaface_loso_3":
        label_dict = {'positive': 0, 'negative': 1, 'surprise': 2}
        labels = list(label_dict.values())

    final_gt = np.array(final_gt)
    final_pred = np.array(final_pred)
    # Macro F1 and Macro Recall (UF1 / UAR)
    UF1 = f1_score(final_gt, final_pred, labels=labels, average='macro', zero_division=0)
    UAR = recall_score(final_gt, final_pred, labels=labels, average='macro', zero_division=0)
    return UF1, UAR


def extract_prefix(file_name):
    prefixes = ["_1_u", "_2_u", "_1_v", "_2_v", "_apex"]
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
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


def get_loss_function(loss_name, weight=None):
    if loss_name == "CELoss":
        return nn.CrossEntropyLoss()
    elif loss_name == "FocalLoss":
        return FocalLoss()
    elif loss_name == "FocalLoss_weighted":
        return FocalLoss(weight=weight)


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
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = nn.KLDivLoss(reduction="batchmean")(output_log_softmax, target_output)
    return loss_kd


def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()


# 每一类样本数量
CASME2_5_numbers = [32, 25, 63, 27, 99]
CASME2_3_numbers = [32, 90, 25]
SAMM_3_numbers = [26, 92, 15]
# {'happy': 0, 'surprise': 1, 'disgust': 2, 'anger': 3, 'fear': 4, 'sad': 5, 'others': 6}
CASME3_7_numbers = [55, 187, 250, 64, 86, 57, 161]
CASME3_4_numbers = [55, 457, 187, 161]
CASME3_3_numbers = [55, 457, 187]


# # 自动统计
# def get_dataset_numbers():


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
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # numbers初始化 以防报错
    numbers = []
    # dataset_name = os.path.basename(config.main_path).split("_")[0]  # "CASME2_retinaface_loso"
    # 得取全名
    dataset_name = os.path.basename(config.main_path)  # "CASME2_retinaface_loso_5"
    print("数据集名称:")
    # print("test")
    print(dataset_name)

    if config.loss_function == "FocalLoss_weighted":
        # # 原匹配方式
        # if config.main_path.split("/")[1].split("_")[0] == "CASME2":
        #     numbers = CASME2_numbers
        # 自定义匹配方式
        if dataset_name == "CASME2_retinaface_loso_5":
            numbers = CASME2_5_numbers
        elif dataset_name == "CASME2_retinaface_loso_3":
            numbers = CASME2_3_numbers
        elif dataset_name == "SAMM_retinaface_loso_3":
            numbers = SAMM_3_numbers
        elif dataset_name == "CASME3_retinaface_loso_7":
            numbers = CASME3_7_numbers
        elif dataset_name == "CASME3_retinaface_loso_4":
            numbers = CASME3_4_numbers
        elif dataset_name == "CASME3_retinaface_loso_3":
            numbers = CASME3_3_numbers

        sum_reciprocal = sum(1 / num for num in numbers)
        weights = [(1 / num) / sum_reciprocal for num in numbers]

        loss_fn = get_loss_function(config.loss_function, torch.tensor(weights).to(device))
    else:
        loss_fn = get_loss_function(config.loss_function)

    if (config.train):
        if not path.exists('/kaggle/working/Experiment_for_recognize/' + config.exp_name):
            os.makedirs('/kaggle/working/Experiment_for_recognize/' + config.exp_name)

    current_file = os.path.abspath(__file__)
    shutil.copy(current_file, '/kaggle/working/Experiment_for_recognize/' + config.exp_name)
    shutil.copy(all_model_path, '/kaggle/working/Experiment_for_recognize/' + config.exp_name)

    log_file_path = '/kaggle/working/Experiment_for_recognize/' + config.exp_name + "/logs.txt"
    sys.stdout = Logger(log_file_path)

    total_gt = []
    total_pred = []
    best_total_pred = []
    all_accuracy_dict = {}

    t = time.time()

    main_path = config.main_path
    subName = os.listdir(main_path)
    # 由于数据集太大 分割
    # 不再使用
    # if config.part_Subjects:
    #     subName = ast.literal_eval(config.part_Subjects)
    #     # 接着上一次的
    #     total_gt = ast.literal_eval(config.part_total_gt)
    #     total_pred = ast.literal_eval(config.part_total_pred)
    #     best_total_pred = ast.literal_eval(config.part_best_total_pred)
    #     all_accuracy_dict = ast.literal_eval(config.part_all_accuracy_dict)
    # else:
    #     subName = os.listdir(main_path)
    print("main_path:", main_path)
    print("exists:", os.path.exists(main_path))
    print("listdir:", os.listdir(main_path))
    if len(subName):
        print(subName)
    else:
        print(f"路径为{main_path}的数据集解析错误")
    # 训练给定的subject
    # 特别对于CAS(ME)^3这种大型数据集
    # 加进度条
    for n_subName in tqdm(subName, desc="Subjects"):
    # for n_subName in subName:
        print('Subject:', n_subName)

        X_train = []
        y_train = []

        X_test = []
        y_test = []

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
                            block = large_S[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]

                            scaled_block = cv2.resize(block, (48, 48))

                            end_input.append(scaled_block)

                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size

                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S_onset[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]

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
                            block = large_S[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]

                            scaled_block = cv2.resize(block, (48, 48))

                            end_input.append(scaled_block)

                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size

                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S_onset[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]

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

        writer = SummaryWriter(log_path)

        model = get_model(config.model, config.class_num, config.Aug_alpha).to(device)

        if (config.train):
            if (config.pre_trained):
                model.apply(reset_weights)
                pre_trained_model = torch.load(config.pre_trained_model_path)
                filtered_dict = OrderedDict((k, v) for k, v in pre_trained_model.items() if (not "fc" in k))
                model.load_state_dict(filtered_dict, strict=False)
            elif (config.Aug_COCO_pre_trained):
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
            model.load_state_dict(torch.load(weight_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.0005)
        """
        /kaggle/working/MERecognition/train_classify_SKD_TSTSAN_functions.py:356: 
        UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. 
        Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
         (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:254.)
        X_train = torch.Tensor(X_train).permute(0, 3, 1, 2)
        这个警告是 PyTorch 给你的性能提示，不会导致程序报错，但说明你的做法效率很低
        """
        # X_train = torch.Tensor(X_train).permute(0, 3, 1, 2)
        # y_train = torch.Tensor(y_train).to(dtype=torch.long)
        # dataset_train = TensorDataset(X_train, y_train)
        # X_train 原本是 list of np.ndarray
        X_train = np.array(X_train, dtype=np.float32)  # 转成统一 numpy array
        X_train = torch.from_numpy(X_train).permute(0, 3, 1, 2)  # 转成 Tensor 并调整通道
        # y_train 原本是 list
        y_train = torch.tensor(y_train, dtype=torch.long)
        dataset_train = TensorDataset(X_train, y_train)

        def worker_init_fn(worker_id):
            random.seed(seed + worker_id)
            np.random.seed(seed + worker_id)

        train_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)

        # X_test = torch.Tensor(X_test).permute(0, 3, 1, 2)
        # y_test = torch.Tensor(y_test).to(dtype=torch.long)
        # dataset_test = TensorDataset(X_test, y_test)
        # X_test 原本是 list of np.ndarray
        X_test = np.array(X_test, dtype=np.float32)
        X_test = torch.from_numpy(X_test).permute(0, 3, 1, 2)  # 调整通道顺序
        # y_test 原本是 list
        y_test = torch.tensor(y_test, dtype=torch.long)
        dataset_test = TensorDataset(X_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        max_iter = config.max_iter
        iter_num = 0
        # 轮数
        # max_iter为20000
        epochs = max_iter // len(train_dl) + 1

        for epoch in tqdm(range(1, epochs + 1), desc=f"{n_subName} Epoch"):
        # for epoch in range(1, epochs + 1):
            if (config.train):
                model.train()
                train_ce_loss = 0.0
                middle_loss1 = 0.0
                middle_loss2 = 0.0
                KL_loss1 = 0.0
                KL_loss2 = 0.0
                L2_loss1 = 0.0
                L2_loss2 = 0.0
                loss_sum = 0.0
                proto_loss_sum = 0.0

                num_train_correct = 0
                num_train_examples = 0

                middle1_num_train_correct = 0
                middle2_num_train_correct = 0

                for batch in train_dl:
                    optimizer.zero_grad()
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    yhat, AC1_out, AC2_out, final_feature, AC1_feature, AC2_feature = model(x)
                    # ⭐ Prototype Loss（新增）
                    loss_proto = prototype_loss(final_feature, y, model.prototypes)
                    loss = loss_fn(yhat, y)
                    AC1_loss = loss_fn(AC1_out, y)
                    AC2_loss = loss_fn(AC2_out, y)
                    #
                    temperature = config.temperature
                    temp4 = yhat / temperature
                    temp4 = torch.softmax(temp4, dim=1)
                    loss1by4 = new_kd_loss_function(AC1_out, temp4.detach(), temperature) * (temperature ** 2)
                    loss2by4 = new_kd_loss_function(AC2_out, temp4.detach(), temperature) * (temperature ** 2)
                    feature_loss_1 = feature_loss_function(AC1_feature, final_feature.detach())
                    feature_loss_2 = feature_loss_function(AC2_feature, final_feature.detach())

                    # total_losses = loss + (1 - config.alpha) * (AC1_loss + AC2_loss) + \
                    #                config.alpha * (loss1by4 + loss2by4) + \
                    #                config.beta * (feature_loss_1 + feature_loss_2)
                    # ⭐ 新增权重 lambda_proto（建议0.3~0.5）
                    lambda_proto = config.lambda_proto if hasattr(config, "lambda_proto") else 0.5

                    total_losses = loss + (1 - config.alpha) * (AC1_loss + AC2_loss) + \
                                   config.alpha * (loss1by4 + loss2by4) + \
                                   config.beta * (feature_loss_1 + feature_loss_2) + \
                                   lambda_proto * loss_proto  # ⭐ 加在这里
                    total_losses.backward()
                    optimizer.step()

                    proto_loss_sum += loss_proto.item() * x.size(0)
                    train_ce_loss += loss.data.item() * x.size(0)
                    middle_loss1 += AC1_loss.data.item() * x.size(0)
                    middle_loss2 += AC2_loss.data.item() * x.size(0)
                    KL_loss1 += loss1by4.data.item() * x.size(0)
                    KL_loss2 += loss2by4.data.item() * x.size(0)
                    L2_loss1 += feature_loss_1.data.item() * x.size(0)
                    L2_loss2 += feature_loss_2.data.item() * x.size(0)
                    # 避免 GPU tensor 泄漏
                    # loss_sum += total_losses * x.size(0)
                    loss_sum += total_losses.item() * x.size(0)

                    num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]

                    middle1_num_train_correct += (torch.max(AC1_out, 1)[1] == y).sum().item()
                    middle2_num_train_correct += (torch.max(AC2_out, 1)[1] == y).sum().item()

                    iter_num += 1
                    if iter_num >= max_iter:
                        break

                train_acc = num_train_correct / num_train_examples
                middle1_acc = middle1_num_train_correct / num_train_examples
                middle2_acc = middle2_num_train_correct / num_train_examples

                train_ce_loss = train_ce_loss / len(train_dl.dataset)
                middle_loss1 = middle_loss1 / len(train_dl.dataset)
                middle_loss2 = middle_loss2 / len(train_dl.dataset)
                KL_loss1 = KL_loss1 / len(train_dl.dataset)
                KL_loss2 = KL_loss2 / len(train_dl.dataset)
                L2_loss1 = L2_loss1 / len(train_dl.dataset)
                L2_loss2 = L2_loss2 / len(train_dl.dataset)
                loss_sum = loss_sum / len(train_dl.dataset)
                proto_loss_sum = proto_loss_sum / len(train_dl.dataset)

                writer.add_scalar("Train_Acc", train_acc, epoch)
                writer.add_scalar("Middle1_Train_Acc", middle1_acc, epoch)
                writer.add_scalar("Middle2_Train_Acc", middle2_acc, epoch)
                writer.add_scalar("train_ce_loss", train_ce_loss, epoch)
                writer.add_scalar("middle_loss1", middle_loss1, epoch)
                writer.add_scalar("middle_loss2", middle_loss2, epoch)
                writer.add_scalar("KL_loss1", KL_loss1, epoch)
                writer.add_scalar("KL_loss2", KL_loss2, epoch)
                writer.add_scalar("L2_loss1", L2_loss1, epoch)
                writer.add_scalar("L2_loss2", L2_loss2, epoch)
                writer.add_scalar("proto_loss", proto_loss_sum, epoch)
                writer.add_scalar("loss_sum", loss_sum, epoch)

                writer.add_scalar("Aug Factor", model.amp_factor, epoch)

            model.eval()
            num_val_correct = 0

            middle1_num_val_correct = 0
            middle2_num_val_correct = 0

            num_val_examples = 0
            temp_best_each_subject_pred = []
            temp_y = []
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

            val_acc = num_val_correct / num_val_examples
            middle1_val_acc = middle1_num_val_correct / num_val_examples
            middle2_val_acc = middle2_num_val_correct / num_val_examples

            writer.add_scalar("Val_Acc", val_acc, epoch)
            writer.add_scalar("Middle1_Val_Acc", middle1_val_acc, epoch)
            writer.add_scalar("Middle2_Val_Acc", middle2_val_acc, epoch)
            if best_accuracy_for_each_subject <= val_acc:
                best_accuracy_for_each_subject = val_acc
                best_each_subject_pred = temp_best_each_subject_pred
                if (config.train) and (config.save_model):
                    torch.save(model.state_dict(), weight_path)

            if val_acc == 1:
                break

            if not (config.train):
                break

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
        # config.main_path.split("/")[1].split("_")[0] 原先指的是数据集的名称 CASME2
        # main_path=/kaggle/working/CASME2_retinaface_loso
        UF1, UAR = recognition_evaluation(dataset_name, total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(dataset_name, total_gt,
                                                    best_total_pred, show=True)
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    writer.close()
    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(dataset_name, total_gt, total_pred, show=True)
    best_UF1, best_UAR = recognition_evaluation(dataset_name, total_gt,
                                                best_total_pred, show=True)
    print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
    print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))
    # 数据集太大 进行分割
    # 不再使用
    # if config.part_Subjects:
    #     print("================================================")
    #     print("可初始化为下一次训练的数据：")
    #     print(f"total_gt: {total_gt}")
    #     print(f"total_pred: {total_pred}")
    #     print(f"best_total_pred: {best_total_pred}")
    #     print(f"all_accuracy_dict: {all_accuracy_dict}")
    #     print("================================================")
    elapsed = time.time() - t
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"Total Time Taken: {hours}hours {minutes}minutes {seconds}seconds")
    print(all_accuracy_dict)

    sys.stdout.log.close()
