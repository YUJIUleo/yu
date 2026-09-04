"""
第二步：GCN对比学习模型训练
运行方式：python train.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互模式，直接保存图片
import matplotlib.pyplot as plt
import os

# ==================== 超参数 ====================
EPOCHS = 30
LR = 0.01
BATCH_SIZE = 16
EMBED_DIM = 16
TEMPERATURE = 0.5
DROP_RATE = 0.15  # 节点丢弃率

# ==================== 设备选择 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

# ==================== 加载数据 ====================
all_data = torch.load("data/all_graphs.pt", weights_only=False)
print(f"加载了 {len(all_data)} 个图数据")

# 划分训练/测试集（80%训练，20%测试）
np.random.seed(42)
indices = np.random.permutation(len(all_data))
split = int(len(all_data) * 0.8)
train_idx, test_idx = indices[:split], indices[split:]

train_data = [all_data[i] for i in train_idx]
test_data = [all_data[i] for i in test_idx]
print(f"训练集: {len(train_data)} | 测试集: {len(test_data)}")

# ==================== GCN编码器 ====================
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index)
        h = global_mean_pool(h, batch)  # 全局平均池化，得到图级嵌入
        return F.normalize(h, p=2, dim=-1)  # L2归一化

model = GCNEncoder(in_dim=3, hidden_dim=32, out_dim=EMBED_DIM).to(device)
print(f"\n模型结构:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params}")

# ==================== 数据增强（节点丢弃） ====================
def augment_graph(data, drop_rate=0.15):
    """随机丢弃部分节点特征"""
    x = data.x.clone()
    mask = torch.rand(x.size(0)) > drop_rate
    x[~mask] = torch.zeros_like(x[~mask])
    return data.__class__(x=x, edge_index=data.edge_index.clone(), y=data.y.clone())

# ==================== InfoNCE对比损失 ====================
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        # z1, z2: [batch_size, embed_dim]
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0)).to(z1.device)
        loss = F.cross_entropy(sim, labels)
        return loss

criterion = InfoNCELoss(temperature=TEMPERATURE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ==================== 训练循环 ====================
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
loss_history = []

print("\n开始训练...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        
        # 同一个batch做两次不同的增强
        aug1_list = [augment_graph(d, DROP_RATE) for d in batch.to_data_list()]
        aug2_list = [augment_graph(d, DROP_RATE) for d in batch.to_data_list()]
        
        from torch_geometric.data import Batch
        batch1 = Batch.from_data_list(aug1_list).to(device)
        batch2 = Batch.from_data_list(aug2_list).to(device)
        
        # 前向传播
        z1 = model(batch1.x, batch1.edge_index, batch1.batch)
        z2 = model(batch2.x, batch2.edge_index, batch2.batch)
        
        # 计算对比损失
        loss = criterion(z1, z2)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f}")

print("✅ 训练完成！")

# 保存模型
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/gcn_encoder.pth")
print("模型已保存到 models/gcn_encoder.pth")

# ==================== 保存损失曲线 ====================
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), loss_history, 'b-o', markersize=4)
plt.xlabel('Epoch')
plt.ylabel('InfoNCE Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/loss_curve.png", dpi=150)
print("损失曲线已保存到 outputs/loss_curve.png")

# ==================== 提取嵌入并评估 ====================
model.eval()
test_loader = DataLoader(test_data, batch_size=len(test_data))

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        embeddings = model(batch.x, batch.edge_index, batch.batch)
        labels = batch.y.cpu().numpy()

embeddings_np = embeddings.cpu().numpy()

# KNN分类评估
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(embeddings_np, labels)
preds = knn.predict(embeddings_np)

print("\n分类评估报告:")
target_names = ['HIP(锥体)', 'DG(颗粒)', 'CB(浦肯野)']
print(classification_report(labels, preds, target_names=target_names))

# 混淆矩阵
cm = confusion_matrix(labels, preds)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['HIP', 'DG', 'CB'])
ax.set_yticklabels(['HIP', 'DG', 'CB'])
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
plt.colorbar(im)
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/confusion_matrix.png", dpi=150)
print("混淆矩阵已保存到 outputs/confusion_matrix.png")

# 保存嵌入用于可视化
np.save("outputs/embeddings.npy", embeddings_np)
np.save("outputs/labels.npy", labels)
print("嵌入向量已保存到 outputs/embeddings.npy")
print("\n🎉 全部训练和评估完成！")