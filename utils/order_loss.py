import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





# 这个函数的目的是将真实标签转化为软变迁
# 生成一个定制化的损失函数，它考虑到了类别之间的距离或者差异,有序数回归问题
def true_metric_loss(true, no_of_classes, scale=1):
    # true: 真实的标签向量，其中每个元素代表一个样本的类别标签。
    # no_of_classes: 数据集中总的类别数量。
    # scale: 一个可选的缩放因子，默认值为1，用于调节类别之间差异的影响。
    batch_size = true.size(0) # 批次中样本的数量。

    true = true.view(batch_size,1) # 将真实标签向量转换为(batch_size, 1)的形状。
    # 将真实标签向量转换为LongTensor类型，并在列方向上重复no_of_classes次，形成一个矩阵，然后转换为浮点数。这个矩阵的每一行都是相同的真实标签。
    true_labels = torch.cuda.LongTensor(true).repeat(1, no_of_classes).float()
    # class_labels = torch.arange(no_of_classes).float().cuda()：生成一个从0到no_of_classes-1的连续整数向量，然后转换为浮点数并移动到CUDA设备上。
    class_labels = torch.arange(no_of_classes).float().cuda()
    # 计算class_labels向量和true_labels矩阵之间的绝对差值，然后乘以缩放因子scale
    phi = (scale * torch.abs(class_labels - true_labels)).cuda()
    # 对phi矩阵的每一行进行softmax操作，得到一个概率分布。
    y = nn.Softmax(dim=1)(-phi) # 用-phi是为了让距离较小（即类别接近真实标签）的类别有较高的概率值。
    return y


def loss_function(output, labels, loss_type, expt_type=5, scale=1.8):
    targets = true_metric_loss(labels, expt_type, scale)
    return torch.sum(- targets * F.log_softmax(output, -1), -1).mean()


def gr_metrics(op, t):
    op = np.array(op)  # 确保 op 是 NumPy 数组
    t = np.array(t)    # 确保 t 是 NumPy 数组
    # 
    TP = (op==t).sum()
    # FN（False Negative）：实际为正例，但预测为负例的数量。
    FN = (t>op).sum()
    # FP（False Positive）：实际为负例，但预测为正例的数量。
    FP = (t<op).sum()

    GP = TP/(TP + FP)
    GR = TP/(TP + FN)

    FS = 2 * GP * GR / (GP + GR)

    # 过估计错误率（OE, Overestimation Error）：

    OE = (t-op > 1).sum() # 计算预测值与真实标签之差大于1的次数，即模型严重过估计的情况。
    OE = OE / op.shape[0] # 过估计错误率。

    return GP, GR, FS, OE