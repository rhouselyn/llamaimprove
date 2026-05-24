import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.distributions as dist
import csv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 可能是由于是MacOS系统的原因

### 本例实现了用神经网络计算互信息的功能。这是一个简单的例子，目的在于帮助读者更好地理解MINE方法。
### 修改说明：
### - 去除未使用的函数：删除 gen_x(), gen_y(), show_data() 和 data_size，只保留核心训练逻辑。
### - 数据生成：使用 Bernoulli(0.5) 生成 0/1 向量（8维），y 以 80% 概率保持 x 值，20% 翻转，引入相关性。
### - 数值稳定性：添加 clamp(pred_x_y, min=-20, max=20) 防止 exp 溢出，避免 -inf。
### - 训练后保存权重到 './mine_weights.pth'。
### - 每个 epoch 保存 MI 值图（loss_plot.png）和数据（plot_loss.npy, epoch_loss.csv）。
### - 新增：添加独立样本计算，生成16维Bernoulli向量，打乱维度后分割左右8维，再打乱右边行计算ret_indep。
### - 损失：- (ret + ret_indep)，同时训练让独立趋向0，相关变大。
### - 修复：shuffle逻辑改为全样本打乱（平铺后shuffle）。
### - 新增：绘制两条曲线（相关 MI 和独立下界），保存为字典 npy 和三列 CSV。
### - 新增：加入 EMA 机制以稳定边缘期望，并使用代理损失 surrogate 优化梯度，减少方差。
### - 修改：简化独立样本生成，直接生成三个独立的8维Bernoulli向量作为 left, right, right_shuffle，效果等价于原逻辑（因为 iid 分布，生成新样本 ≈ 大样本 shuffle）。
### - 新增：为绘图添加第三条曲线，使用原shuffle打乱方法计算的独立MI（ret_indep_shuffle），仅用于计算和绘图，不参与loss或EMA（保持原loss不变）。

def gen_x_tensor():
    # 使用 Bernoulli(0.5) 生成 0/1 向量
    return dist.Bernoulli(probs=0.5).sample((128, 96, 8))

def gen_y_tensor(x):
    # 引入相关性：以概率 0.8 保持 x 的值，否则翻转 (1-x)
    flip_prob = 0.2  # 翻转概率，控制相关强度
    flip_mask = dist.Bernoulli(probs=flip_prob).sample(x.shape)
    y = x * (1 - flip_mask) + (1 - x) * flip_mask
    return y

def gen_indep_tensor():
    # 生成独立16维Bernoulli向量（用于原shuffle方法）
    return dist.Bernoulli(probs=0.5).sample((128, 96, 16))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x) + self.fc2(y))
        h2 = self.fc3(h1)
        return h2

if __name__ == '__main__':
    model = Net()  # 实例化模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 使用 Adam 优化器并设置学习率为 0.01

    ma_et = None
    ma_et_indep = None
    alpha = 0.99  # EMA coefficient for the old value

    n_epoch = 2000
    plot_loss_correlated = []  # 相关 MI (-ret)
    plot_loss_indep = []  # 独立下界 (ret_indep, 直接生成方法)
    plot_loss_indep_shuffle = []  # 独立下界 (ret_indep_shuffle, 原shuffle方法，仅用于绘图)
    for epoch in tqdm(range(n_epoch)):
        # 相关样本
        x_sample = gen_x_tensor()  # 生成样本 x_sample，代表 X 的边缘分布 P(X)
        y_sample = gen_y_tensor(x_sample)  # 生成样本 y_sample，代表条件分布 P(Y|X)

        # 全样本打乱 y (平铺后shuffle)
        flat_num = x_sample.shape[0] * x_sample.shape[1]  # 128 * 96
        perm = torch.randperm(flat_num)
        y_flat = y_sample.view(flat_num, 8)
        y_shuffle_flat = y_flat[perm]
        y_shuffle = y_shuffle_flat.view(y_sample.shape)

        # 独立样本：直接生成三个独立的8维向量，用于loss计算
        left = dist.Bernoulli(probs=0.5).sample((128, 96, 8))
        right = dist.Bernoulli(probs=0.5).sample((128, 96, 8))
        right_shuffle = dist.Bernoulli(probs=0.5).sample((128, 96, 8))

        # 额外：使用原shuffle方法生成独立样本，仅用于计算ret_indep_shuffle（不用于loss）
        indep_sample = gen_indep_tensor()  # (128, 96, 16)
        perm_dim = torch.randperm(16)
        indep_perm = indep_sample[..., perm_dim]
        left_shuffle = indep_perm[..., :8]
        right_orig = indep_perm[..., 8:]
        # 全样本打乱 right_orig
        perm_shuffle = torch.randperm(flat_num)
        right_flat = right_orig.view(flat_num, 8)
        right_shuffle_old = right_flat[perm_shuffle].view(right_orig.shape)

        model.zero_grad()

        # 相关计算
        pred_xy = model(x_sample, y_sample)  # 联合分布的期望
        pred_x_y = model(x_sample, y_shuffle)  # 边缘分布的期望
        pred_x_y = torch.clamp(pred_x_y, min=-20, max=20)
        et = torch.mean(torch.exp(pred_x_y))
        if ma_et is None:
            ma_et = et.detach()
        ret = torch.mean(pred_xy) - torch.log(et + 1e-8)  # 互信息估计（用于报告，无偏）
        surrogate = torch.mean(pred_xy) - (et / ma_et)  # 代理损失（用于优化，低方差梯度）

        # 独立计算（直接生成方法，用于loss）
        pred_xy_indep = model(left, right)
        pred_x_y_indep = model(left, right_shuffle)
        pred_x_y_indep = torch.clamp(pred_x_y_indep, min=-20, max=20)
        et_indep = torch.mean(torch.exp(pred_x_y_indep))
        if ma_et_indep is None:
            ma_et_indep = et_indep.detach()
        ret_indep = torch.mean(pred_xy_indep) - torch.log(et_indep + 1e-8)  # 独立互信息估计，应趋向0
        surrogate_indep = torch.mean(pred_xy_indep) - (et_indep / ma_et_indep)

        # 额外独立计算（原shuffle方法，仅用于绘图）
        pred_xy_indep_shuffle = model(left_shuffle, right_orig)
        pred_x_y_indep_shuffle = model(left_shuffle, right_shuffle_old)
        pred_x_y_indep_shuffle = torch.clamp(pred_x_y_indep_shuffle, min=-20, max=20)
        ret_indep_shuffle = torch.mean(pred_xy_indep_shuffle) - torch.log(torch.mean(torch.exp(pred_x_y_indep_shuffle)) + 1e-8)

        loss = - surrogate #+ 2*torch.abs(surrogate_indep + 1)  # 最大化 surrogate（相关MI），最小化 |surrogate_indep + 1|（独立趋-1调整为0）
        plot_loss_correlated.append(ret.item())  # 相关 MI 值（正向增长）
        plot_loss_indep.append(ret_indep.item())  # 独立下界值（直接生成方法，负向0）
        plot_loss_indep_shuffle.append(ret_indep_shuffle.item())  # 独立下界值（原shuffle方法，负向0）

        loss.backward()  # 反向传播
        optimizer.step()  # 更新优化器

        # 更新 EMA（仅用于直接生成方法）
        ma_et = alpha * ma_et + (1 - alpha) * et.detach()
        ma_et_indep = alpha * ma_et_indep + (1 - alpha) * et_indep.detach()

        # 每个 epoch 保存图片和数据（覆盖）
        epochs = np.arange(len(plot_loss_correlated))
        plt.plot(epochs, plot_loss_correlated, 'r', label='Correlated MI (-ret)')
        plt.plot(epochs, plot_loss_indep, 'b', label='Independent Bound (direct)')
        plt.plot(epochs, plot_loss_indep_shuffle, 'g', label='Independent Bound (shuffle)')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('MINE Training Progress: Correlated vs Independent (Direct vs Shuffle)')
        plt.legend()
        plt.savefig('loss_plot_16.png', bbox_inches='tight')  # 保存图片，覆盖
        plt.close()  # 关闭当前 figure，避免叠加

        # 保存 npy：字典形式
        np.save('plot_loss_16.npy', {'correlated': np.array(plot_loss_correlated), 'indep_direct': np.array(plot_loss_indep), 'indep_shuffle': np.array(plot_loss_indep_shuffle)})

        # 保存 CSV：epoch, mi_correlated, mi_indep_direct, mi_indep_shuffle，覆盖整个文件（重新写入累计数据）
        with open('epoch_loss.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'mi_correlated', 'mi_indep_direct', 'mi_indep_shuffle'])
            for i in range(len(plot_loss_correlated)):
                writer.writerow([i, plot_loss_correlated[i], plot_loss_indep[i], plot_loss_indep_shuffle[i]])

    # 训练结束后保存模型权重
    torch.save(model.state_dict(), './mine_weights_16.pth')
    print("MINE model weights saved to './mine_weights.pth'")

    # 测试完全独立数据中的 MINE 值平均值
    num_tests = 100  # 测试次数，计算平均值
    mi_values = []
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 无需梯度计算
        for _ in range(num_tests):
            left = dist.Bernoulli(probs=0.5).sample((128, 96, 8))
            right = dist.Bernoulli(probs=0.5).sample((128, 96, 8))
            right_shuffle = dist.Bernoulli(probs=0.5).sample((128, 96, 8))

            pred_xy_indep = model(left, right)
            pred_x_y_indep = model(left, right_shuffle)
            pred_x_y_indep = torch.clamp(pred_x_y_indep, min=-20, max=20)
            et_indep = torch.mean(torch.exp(pred_x_y_indep))
            ret_indep = torch.mean(pred_xy_indep) - torch.log(et_indep + 1e-8)
            mi_values.append(ret_indep.item())

    avg_mi = sum(mi_values) / num_tests
    with open('independent_mi_avg.txt', 'w') as f:
        f.write(f"Average MINE value on completely independent data: {avg_mi}\n")
    print(f"Saved independent MINE average to 'independent_mi_avg.txt': {avg_mi}")
