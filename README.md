# mlp2gpt

> 从一个 10 万参数的 MLP 开始,一路手写进化到自己的小参数 LLM。

这是一个"以代码为主线"的 LLM 学习仓库:按照 [llm-learning-roadmap.md](./llm-learning-roadmap.md) 的规划,从全连接神经网络(MLP)出发,逐阶段亲手实现、训练、进化,最终目标是拥有一个自己从零写出的小参数 GPT,并能读懂以 DeepSeek 为代表的现代旗舰架构。

每个阶段只交付三样东西:**能训的代码、能看的实验结果、能讲清的文章**。检验标准永远是路线图里的那句话——**改得动、训得出、讲得清**,而不是"看懂了"。

## 路线图与进度

| 阶段 | 目标 | 交付物 | 状态 |
|---|---|---|---|
| **0. 前置基础** | 能用 PyTorch 写训练循环、懂反向传播 | [`mlp_mnist/`](./mlp_mnist):三层 MLP 识别 MNIST 手写数字 | ✅ 已完成 |
| **1. 手写 Dense GPT** | 理解 Transformer 每个组件为什么存在 | 从零实现 attention / FFN / 残差,训出 10M 参数小 GPT 并生成文本 | ⬜ 未开始 |
| **2. Dense 现代化** | 掌握 2024+ 标准 LLM 的每个部件及动机 | RMSNorm / RoPE / GQA / SwiGLU 逐项改造 + 消融对比表 | ⬜ 未开始 |
| **3. MoE** | 理解稀疏激活、负载均衡与专家塌缩 | 8 专家 top-2 MoE + 辅助均衡 loss + 专家利用率分析 | ⬜ 未开始 |
| **4. 精读 DeepSeek 演进线** | 看懂现代旗舰架构的来龙去脉 | V2 → V3 → V3.2 → V4 四份一页纸论文笔记 | ⬜ 未开始 |
| **5. 拓展方向(选一个)** | 建立自己的差异化方向 | 线性注意力 / 推理加速 / 后训练,三选一深入 | ⬜ 未开始 |

完整的周计划、常见坑、过关标准、论文与资源清单见 [llm-learning-roadmap.md](./llm-learning-roadmap.md)。

## 仓库结构

```
mlp2gpt/
├── llm-learning-roadmap.md     # 完整学习路线图(总览表 + 阶段详解 + 论文清单)
├── mlp_mnist/                  # 阶段 0:三层 MLP + MNIST(784→128→64→10)
│   ├── data.py                 #   数据管线:ModelScope MNIST + 预处理
│   ├── model.py                #   模型定义:三层 MLP + ReLU + Dropout
│   ├── train.py                #   训练循环:训练/验证 + 保存最佳模型
│   └── infer.py                #   推理脚本:单张图片预测
├── blogs/                      # 每个阶段配套的博客文章
│   └── 01-mlp-mnist.md         #   从 MLP 到 GPT(一):手写 MLP 识别手写数字
└── models/                     # 训练产出的权重(git 忽略)
```

后续阶段(手写 GPT、MoE 等)会以新的子目录加入,遵循"只维护一个仓库"的原则,让代码的连续性成为理解的连续性。

## 环境依赖

- Python 3.10+
- PyTorch(有 CUDA 显卡最好,纯 CPU 也能跑本仓库目前的实验)
- torchvision、modelscope、pillow

```bash
pip install torch torchvision modelscope pillow
```

## 相关文章

- [从 MLP 到 GPT(一):用 10 万参数的神经网络认出手写数字](./blogs/01-mlp-mnist.md) —— 阶段 0 的完整代码解读:MLP 的数学、数据管线、训练循环四步曲、推理时的坑,以及这一步和 GPT 的关系。
