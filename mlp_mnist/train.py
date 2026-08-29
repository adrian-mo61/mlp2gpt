import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from modelscope import MsDataset

# 导入你的模块（确保 model.py 和 data.py 在同一目录）
from model import MnistModel
from data import MnistDataset, transform

# ------------------- 超参数设置 -------------------
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
DROP_RATE = 0.5
INPUT_SIZE = 28 * 28          # MNIST 图片尺寸
HIDDEN_SIZE = [128, 64]       # 两个隐藏层
OUTPUT_SIZE = 10              # 10 个数字类别

# ------------------- 设备配置 -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------- 加载数据集 -------------------
# 下载训练集（如果已下载则直接加载）
full_dataset = MsDataset.load('mnist', split='train', cache_dir='./datasets')
# 包装成自定义 Dataset
dataset = MnistDataset(full_dataset, transform=transform)

# 按 8:2 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ------------------- 构建模型 -------------------
model = MnistModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, DROP_RATE).to(device)

# ------------------- 损失函数 & 优化器 -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------- 训练循环 -------------------
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # ---------- 训练阶段 ----------
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播 + 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)

    # ---------- 验证阶段 ----------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = correct / total

    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '../models/best_mnist_model.pth')
        print(f"  --> 保存新最佳模型 (Acc: {val_acc:.4f})")

print("训练完成！")