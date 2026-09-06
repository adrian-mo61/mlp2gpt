# mlp_mnist —— 阶段 0:手写三层 MLP 识别 MNIST

> [路线图](../llm-learning-roadmap.md)阶段 0(前置基础)的交付物:用纯 PyTorch 手写训练循环,训练一个三层全连接网络(MLP)完成 MNIST 手写数字分类,并支持对单张图片进行推理。

不依赖任何"一键训练"的高级封装,数据管线、模型、训练循环、推理全部手写,目的是把神经网络最基本的心智模型打牢:**前向传播算出 loss,反向传播算出梯度,优化器更新参数**——这套循环会一直用到后面的 GPT。

## 模型结构

把 28×28 的图片拉平成 784 维向量,经过两个隐藏层,输出 10 个类别的 logits:

```
输入 (28×28=784)          隐藏层 1         隐藏层 2        输出
   [1×784] ──flatten──► Linear(784,128) ──► Linear(128,64) ──► Linear(64,10)
                            │  ReLU            │  ReLU            (logits)
                            ▼ Dropout(0.5)     ▼ Dropout(0.5)
```

参数量:784×128+128 + 128×64+64 + 64×10+10 = **109,386(约 10.9 万)**,与路线图里"10 万参数的 MLP"相符。

## 文件说明

| 文件 | 内容 |
|---|---|
| `data.py` | 数据管线:从 ModelScope 加载 MNIST,包装成自定义 `Dataset`,定义预处理(`ToTensor` + MNIST 均值方差归一化) |
| `model.py` | `MnistModel`:三层 `nn.Linear` + ReLU + Dropout,`forward` 返回 logits |
| `train.py` | 完整训练脚本:超参数、8:2 训练/验证划分、训练循环、按验证准确率保存最佳模型 |
| `infer.py` | 推理:加载权重,读取任意图片(自动转灰度),输出预测数字与置信度 |

## 快速开始

```bash
# 1. 安装依赖
pip install torch torchvision modelscope pillow

# 2. 在仓库根目录创建权重输出目录(train.py 会保存到 ../models/,目录不存在会报错)
mkdir ../models    # 在 mlp_mnist/ 下执行;或在任何位置: mkdir <仓库根>/models

# 3. 训练(首次运行会自动从 ModelScope 下载 MNIST 到 ./datasets)
python train.py

# 4. 用训练好的权重推理单张图片(修改 infer.py 底部的 img_path / model_path)
python infer.py
```

> 推理时 `model_path` 要指向实际权重文件。训练保存在 `<仓库根>/models/best_mnist_model.pth`,而 `infer.py` 默认读当前目录下的 `best_mnist_model.pth`,记得传对路径。

## 超参数

| 超参数 | 值 | 说明 |
|---|---|---|
| batch_size | 64 | |
| epochs | 5 | MNIST 上 MLP 收敛很快,5 轮足够 |
| learning_rate | 1e-3 | Adam 的常用起点 |
| optimizer | Adam | 自适应学习率,调参成本低 |
| loss | CrossEntropyLoss | 内部含 LogSoftmax,故模型末层**不**加 softmax |
| dropout | 0.5 | 两个隐藏层后各接一次,仅训练时生效 |
| 数据划分 | 8:2 | 60000 张训练图 → 48000 训练 / 12000 验证 |

## 结果

按上述配置训练 5 个 epoch,验证集准确率通常在 **97% ~ 98%**(这是该规模 MLP 在 MNIST 上的正常水平)。每次训练的日志留档如下(可自行补充):

```
Epoch 1/5 | Train Loss: 0.___ | Val Loss: 0.___ | Val Acc: 0.9___
...
```

> 注意:`train.py` 只加载了 MNIST 的 `train` split 并按 8:2 划分,官方 10000 张的 `test` split 并未使用。这里的"验证集"实际承担了测试集的角色;更严格的做法是留出 test split 做最终评估,后续阶段会补上。

## 设计要点与注意事项

- **归一化常数**:`Normalize((0.1307,), (0.3081,))` 中的 0.1307 / 0.3081 是整个 MNIST 训练集的像素均值 / 标准差,把输入拉到零均值附近,收敛更稳更快。
- **自定义 Dataset 三件套**:`__init__` / `__len__` / `__getitem__`,把 ModelScope 的 `MsDataset` 适配成 PyTorch `DataLoader` 需要的接口。
- **train/eval 模式切换**:Dropout 只在 `model.train()` 下随机置零,验证前必须 `model.eval()` 并配 `torch.no_grad()`,否则评估结果被噪声污染。
- **保存最佳模型**:每个 epoch 结束按验证准确率判断是否覆盖 `best_mnist_model.pth`,天然带 early-stopping 效果。
- **推理预处理必须与训练完全一致**:同一份 `transform`、同样转灰度(`convert('L')`)。另外 MNIST 是**白字黑底**,如果拿黑字白底的图片(如白纸黑笔拍照)直接预测,效果会很差,需要先反色。
- **可复现性**:`random_split` 未固定随机种子,每次训练的验证集划分不同;要严格对齐结果可加 `generator=torch.Generator().manual_seed(42)`。
- **Windows 用户**:`DataLoader(num_workers=2)` 在 Windows 上要求主脚本有 `if __name__ == "__main__":` 保护,`train.py` 目前没有,若在 Windows 报多进程错误,把 `num_workers` 改为 0 或加保护即可。

## 与路线图的衔接

这一步覆盖了路线图阶段 0 的目标:能默写训练循环、理解交叉熵 + 反向传播由 `autograd` 完成。下一步(阶段 1)把"拉平的像素向量"换成"token embedding + attention",复用完全相同的训练循环去训一个手写 GPT。

详细解读见博客:[从 MLP 到 GPT(一)](https://adrian-mo61.github.io/posts/mlp-mnist/)。
