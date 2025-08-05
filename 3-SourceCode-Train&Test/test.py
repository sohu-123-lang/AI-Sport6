import torch
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib.patches as patches

# 定义模型类（与训练时相同）
class MultiModalModel(torch.nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=3, hidden_size=64, num_layers=2, batch_first=True)
        self.fc_team = torch.nn.Linear(2 * 2, 64)
        self.rnn_gesture = torch.nn.LSTM(input_size=12, hidden_size=64, num_layers=2, batch_first=True)
        self.fc_fusion = torch.nn.Linear(64 * 3, 128)
        self.fc_output = torch.nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x1, x2, x3):
        _, (hn, _) = self.rnn(x1)
        x1_out = hn[-1]
        x1_out = self.dropout(x1_out)
        x2_out = F.relu(self.fc_team(x2.view(x2.size(0), -1)))
        _, (hn_gesture, _) = self.rnn_gesture(x3)
        x3_out = hn_gesture[-1]
        x3_out = self.dropout(x3_out)
        combined = torch.cat((x1_out, x2_out, x3_out), dim=1)
        fused = F.relu(self.fc_fusion(combined))
        output = self.fc_output(fused)
        return output

# 加载训练好的模型
model = MultiModalModel()
#$model.load_state_dict(torch.load('multi_modal_model_best2-dropout.pth'))
model.load_state_dict(torch.load('multi_modal_model_best2.pth'))

# 强制启用 Dropout，即使在 eval 模式下
def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

enable_dropout(model)

# 加载测试数据
X1 = torch.load('combined_X1.pt')
X2 = torch.load('combined_X2.pt')
X3 = torch.load('combined_X3.pt')
y_true = torch.load('combined_y.pt')  # 加载真实标签

# 初始化存储所有输出
all_outputs = []

# 对所有样本进行推理
num_samples = X1.shape[0]  # 样本总数
with torch.no_grad():
    for i in range(num_samples):
        x1_test = X1[i].unsqueeze(0)  # 增加 batch 维度
        x2_test = X2[i].unsqueeze(0)
        x3_test = X3[i].unsqueeze(0)
        output = model(x1_test, x2_test, x3_test)
        all_outputs.append(output.numpy())

# 将推理结果转换为 NumPy 数组
outputs = np.concatenate(all_outputs, axis=0)

# 计算预测值和真实标签之间的误差（欧几里得距离）
errors = np.linalg.norm(outputs - y_true.numpy(), axis=1)

# 找到误差最大的 10% 的样本
num_top_errors = int(len(errors) * 0.1)  # 取前 10%
top_error_indices = np.argsort(errors)[-num_top_errors:]

# 输出误差最大的 10% 的样本信息
print("Error indices of the top 10%:")
for idx in top_error_indices:
    print(f"Sample {idx}: Predicted {outputs[idx]}, Ground Truth {y_true[idx]}, Error {errors[idx]}")

# 计算平均误差
mean_error = np.mean(errors)
print(f"\nAverage Error: {mean_error:.4f}")

# 可视化误差分布
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, color='blue', alpha=0.7)
plt.axvline(np.percentile(errors, 90), color='red', linestyle='--', label='Top 10% Threshold')
plt.title("Error Distribution")
plt.xlabel("Euclidean Distance Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()
