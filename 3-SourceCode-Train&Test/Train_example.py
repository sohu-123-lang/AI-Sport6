import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 32
NUM_EPOCHS = 10000
LEARNING_RATE = 0.001
KEYPOINT_FEATURES = 12
HIDDEN_DIM = 64  # 推荐设置的隐藏维度

class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        
        # 处理X1（3D位置）投影到更高维度
        self.x1_projection = nn.Linear(3, HIDDEN_DIM)
        
        # Transformer 处理 X1
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_DIM, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)

        # 处理X2（团队位置）
        self.fc_team = nn.Sequential(
            nn.Linear(4, HIDDEN_DIM),  # 将4维输入映射到64维
            nn.ReLU(),
        )
        
        # 处理 X3（手势数据）
        self.x3_projection = nn.Linear(KEYPOINT_FEATURES, HIDDEN_DIM)  # 映射12维到64维
        self.transformer_gesture_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_DIM, nhead=4)
        self.transformer_gesture = nn.TransformerEncoder(self.transformer_gesture_layer, num_layers=2)
        
        # 处理 X4（分类信息）
        self.fc_additional = nn.Sequential(
            nn.Linear(1, HIDDEN_DIM),
            nn.ReLU(),
        )
        
        # 输出层
        self.fc_fusion = nn.Linear(HIDDEN_DIM * 4, 128)
        self.fc_output = nn.Linear(128, 2)  # 输出: 2D落点
    
    def forward(self, x1, x2, x3, x4):
        # 投影 X1
        x1 = self.x1_projection(x1)  # (batch_size, seq_len, HIDDEN_DIM)
        x1_transformed = self.transformer_encoder(x1.permute(1, 0, 2))  # (seq_len, batch_size, HIDDEN_DIM)
        x1_out = x1_transformed[-1]  # 取最后的输出 (batch_size, HIDDEN_DIM)

        # 处理 X2
        x2_out = self.fc_team(x2)  # (batch_size, HIDDEN_DIM)

        # 投影 X3 并处理
        x3 = self.x3_projection(x3)  # (batch_size, seq_len, HIDDEN_DIM)
        x3_transformed = self.transformer_gesture(x3.permute(1, 0, 2))  # (seq_len, batch_size, HIDDEN_DIM)
        x3_out = x3_transformed[-1]  # 取最后的输出 (batch_size, HIDDEN_DIM)

        # 处理 X4
        x4_out = self.fc_additional(x4.view(-1, 1))  # (batch_size, HIDDEN_DIM)

        # 融合所有输出
        combined = torch.cat((x1_out, x2_out, x3_out, x4_out), dim=1)  # 合并所有的输出
        fused = F.relu(self.fc_fusion(combined))  # 融合层
        output = self.fc_output(fused)  # 最终输出层
        return output

def evaluate_model(model, test_loader):
    model.eval()  # 设置模型为评估模式
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for X1, X2, X3, X4, y in test_loader:
            outputs = model(X1, X2, X3, X4)
            all_outputs.append(outputs)
            all_targets.append(y)
    
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    mae = F.l1_loss(all_outputs, all_targets)  # 计算平均绝对误差
    print(f'评估 MAE: {mae.item():.4f}')
    return all_outputs, all_targets, mae

def train_model(model, criterion, optimizer, train_loader, val_loader, scheduler=None):
    model.train()
    for epoch in range(NUM_EPOCHS):
        for X1, X2, X3, X4, y in train_loader:
            optimizer.zero_grad()  # 清空梯度
            outputs = model(X1, X2, X3, X4)  # 前向传播
            loss = criterion(outputs, y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
        if scheduler:
            scheduler.step()
        
        # 每隔一定轮数评估验证集
        _, _, val_loss = evaluate_model(model, val_loader)
        if (epoch + 1) % 100 == 0:  # 每100个epoch打印一次损失
            print(f'第 {epoch + 1} 轮, 损失: {loss.item():.4f}, 验证损失: {val_loss.item():.4f}')

# 初始化模型、损失函数和优化器
model = MultiModalModel()
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 初始化学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # 每100个epoch将学习率减少一半

# 创建数据集
num_samples = 1656
X1 = torch.load('shuffled_X1.pt')  # 加载合并后的数据
X2 = torch.load('shuffled_X2.pt')
X3 = torch.load('shuffled_X3.pt')
X4 = torch.load('shuffled_X4.pt')

y = torch.load('shuffled_y.pt')

# 创建 TensorDataset 和 DataLoader
dataset = TensorDataset(X1, X2, X3, X4, y)
total_size = len(dataset)
train_size = int(0.9 * total_size)  # 90% 用于训练
val_size = int(0.05 * total_size)    # 5% 用于验证
test_size = total_size - train_size - val_size  # 剩余部分用于测试

# 划分数据集
train_dataset, temp_dataset = random_split(dataset, [train_size, total_size - train_size])
val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

# 创建训练和验证的 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 训练模型
train_model(model, criterion, optimizer, train_loader, val_loader, scheduler)

# 评估模型
evaluate_model(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'multi_modal_model_transformer.pth')
print("模型已保存到 'multi_modal_model_transformer.pth'")
