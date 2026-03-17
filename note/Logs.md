### 20250816

**文件结构设置**

* 运行`get_onset_apex_offset.py`后提取的图片的文件结构
* 运行`optflow_for_classify.py`后提取的图片的文件结构

1）是直接将sub之类的全部按下划线连接构成图片的名称，即只有一层文件结构？

2）还是按照原数据集的结构？

3）在运行`get_onset_apex_offset.py`和`optflow_for_classify.py`时仍然和数据集保持一样的文件结构，只是在按照情绪分类时用于LOSO时使用按文件夹层次命名

### 20250825

直接运行

运行结果为https://www.kaggle.com/code/garlic0000/merecognition中的`version 3`

### 20250826

修改文件`all_model.py`为`all_model_1.py`

增加轻量级空间注意力机制

```python
class SpatialAttention(nn.Module):
    # 轻量 CBAM 空间注意力：avg+max -> 7x7 conv -> sigmoid
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
```

在`self.ECA5=...`之后调用

```python
# 空间注意力机制
        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()
        self.SA3 = SpatialAttention()
        self.SA4 = SpatialAttention()
        self.SA5 = SpatialAttention()
        self.SA6 = SpatialAttention()
        self.SA7 = SpatialAttention()
        self.SA8 = SpatialAttention()
        self.SA9 = SpatialAttention()
        self.SA10 = SpatialAttention()
```

这个调用没有设置通道数，不知是否有效果

在`forward`函数中调用

```python
# 第一阶段 conv1 + SA
        x1 = self.conv1_L(x1)
        x1 = self.bn1_L(x1)
        x1 = self.relu(x1)
        x1 = self.ECA1(x1)
        x1 = self.SA1(x1)  # 第1个空间注意力
        x1 = self.maxpool(x1)

        x2 = self.conv1_S(x2)
        x2 = self.bn1_S(x2)
        x2 = self.relu(x2)
        x2 = self.SA2(x2)  # 第2个空间注意力
        x2 = self.maxpool(x2)

        x3 = self.conv1_T(x3)
        x3 = self.bn1_T(x3)
        x3 = self.relu(x3)
        x3 = self.SA3(x3)  # 第3个空间注意力
        x3 = self.maxpool(x3)

        # AC1 模块卷积 + SA
        AC1_x1 = self.AC1_conv1_L(x1)
        AC1_x1 = self.AC1_bn1_L(AC1_x1)
        AC1_x1 = self.relu(AC1_x1)
        AC1_x1 = self.AC1_ECA1(AC1_x1)
        AC1_x1 = self.SA4(AC1_x1)  # 第4个空间注意力
        AC1_x1 = self.AC1_conv2_L(AC1_x1)
        AC1_x1 = self.AC1_bn2_L(AC1_x1)
        AC1_x1 = self.relu(AC1_x1)
        AC1_x1 = self.AC1_ECA2(AC1_x1)
        AC1_x1 = self.SA5(AC1_x1)  # 第5个空间注意力
        AC1_x1 = self.AC1_pool(AC1_x1)
        AC1_x1_all = AC1_x1.view(AC1_x1.size(0), -1)

        AC1_x2 = self.AC1_conv1_S(x2)
        AC1_x2 = self.AC1_bn1_S(AC1_x2)
        AC1_x2 = self.relu(AC1_x2)
        AC1_x2 = self.SA6(AC1_x2)  # 第6个空间注意力
        AC1_x2 = self.AC1_conv2_S(AC1_x2)
        AC1_x2 = self.AC1_bn2_S(AC1_x2)
        AC1_x2 = self.relu(AC1_x2)
        AC1_x2 = self.SA7(AC1_x2)  # 第7个空间注意力
        AC1_x2 = self.AC1_pool(AC1_x2)
        AC1_x2_all = AC1_x2.view(AC1_x2.size(0), -1)

        AC1_x3 = self.AC1_conv1_T(x3)
        AC1_x3 = self.AC1_bn1_T(AC1_x3)
        AC1_x3 = self.relu(AC1_x3)
        AC1_x3 = self.SA8(AC1_x3)  # 第8个空间注意力
        AC1_x3 = self.AC1_conv2_T(AC1_x3)
        AC1_x3 = self.AC1_bn2_T(AC1_x3)
        AC1_x3 = self.relu(AC1_x3)
        AC1_x3 = self.SA9(AC1_x3)  # 第9个空间注意力
        AC1_x3 = self.AC1_pool(AC1_x3)
        AC1_x3_all = AC1_x3.view(AC1_x3.size(0), -1)
```

```python
# 第二阶段高级卷积 x1
        x1 = self.conv2_L(x1)
        x1 = self.bn2_L(x1)
        x1 = self.relu(x1)
        x1 = self.ECA2(x1)
        x1 = self.conv3_L(x1)
        x1 = self.bn3_L(x1)
        x1 = self.relu(x1)
        x1 = self.ECA3(x1)
        x1 = self.SA10(x1)  # 第10个空间注意力
        x1 = self.avgpool(x1)
```

运行结果为https://www.kaggle.com/code/garlic0000/merecognition中的`all_model_1(version4)`



### 20250827

修改文件`train_classify_SKD_TSTSAN_functions.py`

修改函数`confusionMatrix`为

```python
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
```





