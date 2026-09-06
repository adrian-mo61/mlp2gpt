# 小参数 LLM 自学路线图

> 目标：从零手写并逐步进化一个自己的小参数 LLM，最终读懂现代旗舰架构（以 DeepSeek V4 为终点）。
> 前提假设：每周投入 8~10 小时，总时长约 5~8 个月；全职投入可压缩到 2~3 个月。
> 算力要求：全程一张 8GB 显存的消费级显卡即可，单次训练都是小时级。

---

## 一、总览路线表

| 阶段 | 目标 | 动手实践（交付物） | 关键资源 | 预计耗时 |
|---|---|---|---|---|
| **0. 前置基础** | 能用 PyTorch 写训练循环、懂反向传播 | 手写一个两层 MLP 拟合数据集；不借助教程默写 backprop | [Karpathy: The spelled-out intro to neural networks](https://www.youtube.com/watch?v=VMj-3S1tku0)、[动手学深度学习](https://zh.d2l.ai/) | 2~3 周（已会可跳过） |
| **1. 手写 Dense GPT** | 理解 Transformer 每个组件为什么存在 | 从零实现 attention / FFN / 残差，在莎士比亚或 TinyStories 上训出 10M 模型并生成文本 | [nanoGPT](https://github.com/karpathy/nanoGPT)、[Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)、[minbpe](https://github.com/karpathy/minbpe) | 3~4 周 |
| **2. Dense 现代化** | 掌握 2024+ 标准 LLM 的每个部件及其动机 | 逐项改造并消融对比：LayerNorm→RMSNorm、绝对位置→RoPE、MHA→GQA、GeLU→SwiGLU，再加 QK-norm / logit softcapping | [modded-nanoGPT](https://github.com/KellerJordan/modded-nanoGPT)、[RoPE 论文](https://arxiv.org/abs/2104.09864)、[litgpt](https://github.com/Lightning-AI/litgpt) | 4~6 周 |
| **3. MoE** | 理解稀疏激活、负载均衡与专家塌缩 | 实现 8 专家 top-2 MoE，加辅助均衡 loss + router z-loss，观察专家利用率 | [Mixtral 论文](https://arxiv.org/abs/2401.04088)、[DeepSeek-V3 论文 §2](https://arxiv.org/abs/2412.19437)（auxiliary-loss-free balancing、MTP） | 3~4 周 |
| **4. 精读 DeepSeek 演进线** | 看懂现代旗舰架构的来龙去脉 | 顺序读：V2（MLA，手推矩阵吸收公式）→ V3（FP8、MTP）→ V3.2（DSA）→ V4（CSA+HCA、mHC、Muon）；每篇写一页笔记 | [V2](https://arxiv.org/abs/2405.04434) / [V3.2](https://arxiv.org/abs/2509.01064) / [V4 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)、[DeepSeek 开源仓库](https://github.com/deepseek-ai) | 4~6 周 |
| **5. 拓展方向（选一个深入）** | 建立自己的差异化方向 | ① 线性注意力：读 Qwen3-Next / Gated DeltaNet 并对比 CSA 路线；② 推理加速：复现一次推测解码；③ 后训练：SFT + DPO 微调自己的阶段 2 模型 | [Qwen3-Next 博客](https://qwen.ai/blog?id=5492907f2b2a7b23e8a1f351a17580e1e096c09e&from=research.latest-advancements-list)、[DSpark 论文](https://arxiv.org/abs/2607.05147)、[DPO 论文](https://arxiv.org/abs/2305.18290) | 4~8 周 |

**第 5 阶段怎么选**：三个方向分别通向长上下文架构、推理系统和后训练。目标找工作选 ③（后训练岗位需求目前最多），读研/做研究选 ①，对推理系统感兴趣选 ②。

---

## 二、阶段详解（阶段 1~4）

### 阶段 1：手写 Dense GPT（3~4 周）

**心智模型**：一个 LLM 就是五段流水线——`分词 → embedding → N 层重复的块（注意力 + FFN）→ 输出头 → softmax 采样`。目标：让每一段都从"名词"变成亲手实现过的"过程"。

**周计划**：

- **第 1 周：数据和分词。** 用 minbpe 思路手写 BPE tokenizer（byte-level + BPE merge 即可）。写数据管线：文本切成固定长度 block_size 的样本，`(x, y) = (tokens[:-1], tokens[1:])`。
- **第 2 周：注意力（最核心）。** 顺序：单头 → 因果掩码 → 多头。必须弄明白三件事：
  1. **为什么除以 √d**：d 大时点积方差大，softmax 饱和成 one-hot，梯度消失；
  2. **因果掩码为什么用 `tril` 加 -inf 再 softmax**：未来 token 必须变成 0 概率，直接删行会破坏归一化；
  3. **attention 是"软字典查询"**：Q 查询、K 键、V 值，softmax 是模糊匹配权重。
- **第 3 周：完整块 + 训练循环。** 补齐残差、**pre-norm**（`x = x + Attn(LN(x))`，比 post-norm 稳）、FFN（`4d → GELU → d`）。训练循环：交叉熵、AdamW、梯度裁剪 1.0、余弦调度 + warmup。
- **第 4 周：生成与评估。** 自回归采样（temperature、top-k）。两个 sanity check：
  - 10 个 batch 上过拟合，loss 应趋近 0（不趋近 = 代码有 bug）；
  - 与 bigram 基线对比 loss（应碾压，否则也有 bug）。

**常见坑**：初始化用小方差 0.02 并对残差分支输出乘 1/√(2N)，否则深层训不动；softmax 前减最大值防溢出；显存不够用梯度累积而不是削模型。

**过关标准**：不看教程从空白文件写出能训、能采样、loss 正常的 GPT；能对着图讲清"一个 token 从输入到预测出下一个词"的完整旅程。

---

### 阶段 2：Dense 现代化（4~6 周）

**方法**：逐项把 2018 年 GPT 改成 2024 年 LLaMA 风格，用消融实验证明每项改动的价值。原则：一次只改一个变量、固定随机种子、记录 loss / 显存 / 吞吐三个指标。

| 改造项 | 怎么做 | 应该观察到什么 / 理解什么 |
|---|---|---|
| LayerNorm → **RMSNorm** | 去掉均值中心化，只除以均方根 | 效果基本持平但更省算力——均值中心化大部分是冗余的 |
| 绝对位置 → **RoPE** | 对 Q、K 每两维做位置相关旋转 | 相对位置内建于点积；理解它为什么不能作用于压缩后的 KV——这是读懂 MLA"解耦 RoPE"的伏笔 |
| MHA → **GQA** | KV 头数减到 1/4，Q 头共享 | **训练 loss 几乎不变，但 KV cache 缩到 1/4、推理变快**——关键认知：推理瓶颈是显存带宽不是 FLOPs |
| GeLU → **SwiGLU** | `SwiGLU(x) = (SiLU(xW₁) ⊙ xV)W₂` | 参数量持平需把中间维度从 4d 降到 8d/3；理解门控为什么有效（乘法通道让网络选择性放行信息） |
| 加 **QK-norm** | 对 Q、K 各做一次 RMSNorm | attention logits 最大值不再随训练暴涨——Kimi 的 QK-Clip、V4 的内建 RMSNorm 都在解决这个问题的不同阶段 |
| 用 **Flash Attention / SDPA** | 换成 `F.scaled_dot_product_attention` | 不必会写 kernel，但要能说清显存为什么从 O(N²) 降到 O(N)（不物化完整注意力矩阵） |

**穿插的工程课题**：bf16 混合精度、torch.compile、tied embedding（小模型建议共享）。

**过关标准**：一张消融表（每行一个改动 + loss/显存/吞吐变化）+ 一份"2024 标准配置"代码。消融表比模型更值钱——它证明每个部件你都知道"为什么"。

---

### 阶段 3：MoE（3~4 周）

**核心认知**：MoE 解耦"知识容量"（总参数）和"每 token 计算量"（激活参数）。8 专家 top-2 模型就是这两者的第一次分离实验。

**实现清单（按顺序）**：

1. FFN 换成 MoE 层：router 是 `Linear(d, 8)`，softmax 取 top-2，输出 = 两个专家输出的加权（归一化 gate 值）和。
2. **立刻遇到专家塌缩**（token 全涌向一两个专家）→ 引入**辅助负载均衡 loss**：`L_aux = α · E · Σᵢ fᵢ·Pᵢ`（fᵢ = 专家实际 token 比例，Pᵢ = 平均路由概率），迫使两者趋近 1/E。
3. 不稳定时加 **router z-loss**（惩罚 router logits 的 log-sum-exp），防 softmax 进入数值敏感区。
4. 画**专家利用率直方图**：直观看到"无均衡 → 塌缩 → 加 loss → 均衡但 loss 略涨"全过程。

**此刻精读 DeepSeek-V3 论文 §2**，它回应了你刚踩过的每个坑：
- **auxiliary-loss-free balancing**：给每个专家路由分加 bias，bias 只用于选专家、不参与加权和，按负载误差符号缓慢更新——既均衡又不干扰梯度；
- **细粒度专家**：很多小专家比少量大专家组合更灵活（V3：256 专家选 8 个）；
- **共享专家**：永远激活的 1 个专家学通用模式，避免路由专家重复学。

动手实现简化版 aux-loss-free 路由，对比辅助 loss 方式的差别。

**加分项 MTP**：加一个小块预测 t+2 位置 token。训完自然理解：为什么 MTP 能当推测解码的草稿模型，以及它在推理栈中的位置。

**过关标准**：同激活参数下 MoE loss 优于 dense 基线（或持平且总参数翻倍）；有一组专家利用率随训练变化的图；能讲清 aux-loss-free 与 auxiliary loss 的取舍。

---

### 阶段 4：精读 DeepSeek 演进线（4~6 周）

**读法**：每篇用同一模板做一页纸笔记——「解决什么问题 → 上代为什么不够 → 核心改动（公式级）→ 关键效果数字 → 我能怎么验证」。代码与阅读时间 1:1，别连读四篇。

| 顺序 | 论文 | 必须吃透的点 | 建议的动手练习 |
|---|---|---|---|
| 1 | **DeepSeek-V2**（MLA） | KV cache 是推理硬约束；MLA 把 KV 压缩到低秩 latent（c_KV 512 维 + 解耦 RoPE 的 64 维 key）；**为什么 RoPE 与压缩 KV 不兼容**（位置相关的旋转破坏共享 latent）所以必须解耦；矩阵吸收让解码期不用显式还原 K/V | 手推矩阵吸收的代数过程；制表：1M 上下文下 MHA/GQA/MLA 各需多少 GB KV cache |
| 2 | **DeepSeek-V3** | FP8 混合精度训练（细粒度分块缩放、敏感路径保高精度）；MTP 结构；DualPipe 与 EP 通信重叠——"训练 infra 是第二架构" | 把 671B/37B 与你自己的 MoE 模型放进同一张对比表，校准尺度直觉 |
| 3 | **DeepSeek-V3.2 / DSA** | Lightning Indexer 打分 + 每 query 只算 top-k=2048；复杂度 O(L²)→O(L·k)；**原生稀疏训练**的必要性（稠密模型上硬开稀疏会分布失配）；"注意力即检索"哲学 | 画 DSA 数据流图；算账：k=2048 时 128K 上下文省掉多少注意力计算 |
| 4 | **DeepSeek-V4**（CSA+HCA、mHC、Muon） | CSA（4:1 压缩 + top-k 精读）与 HCA（128:1 压缩 + 全量泛读）为什么互补；滑动窗口与注意力锚点各兜住什么；mHC 用双随机流形约束保证深层稳定；Muon 正交化更新在控制什么（谱范数）；FP4 QAT、OPD 后训练范式 | 画 CSA+HCA 完整数据流图（含滑窗分支和锚点）；在自己阶段 2 的模型上验证 QK-norm 对 logits 爆炸的抑制，呼应 V4 设计 |

**过关标准**：四份笔记能串成连贯叙事——"每一代在回应什么瓶颈"；能不看资料讲清"从 MHA 到 CSA/HCA，KV 的形态经历哪四次变化、各换来什么、付出什么"。做到这点，你就有了读懂未来任何新架构报告的底层框架。

---

## 三、贯穿全程的建议

1. **只维护一个仓库**：让它从阶段 1 的朴素 GPT 一路进化成阶段 3 的 MoE 模型。代码的连续性就是理解的连续性。
2. **最便宜地记录实验**：一个 CSV 或 wandb 免费版，记录每次改动的 loss 曲线。两个月后它是你最好的复习材料。
3. **卡住超过半天就读参考实现**（nanoGPT / litgpt / DeepSeek 开源仓库）的对应函数，对完立刻合上再自己写——不要边看边抄。
4. **检验标准永远是"改得动、训得出、讲得清"**，不是"看懂了"。

---

## 四、背景速查：四大模型架构一页纸

| | 最新旗舰 | 架构关键词 | 注意力方案 | 最突出优势 |
|---|---|---|---|---|
| **Kimi** | K2 / K3（1T→2.8T） | MoE + MuonClip 优化器 | MLA | 开源 Agentic + 极低价格 |
| **DeepSeek** | V4（Pro 1.6T/49B、Flash 284B/13B） | MoE + CSA/HCA 混合注意力 + mHC + Muon | MLA + DSA（V3.2）+ CSA/HCA（V4） | 全栈开源、长上下文成本最低 |
| **GLM** | GLM-5（744B MoE） | MoE + 稀疏注意力 + 异步 RL | DSA 类稀疏注意力 | 工程级 Coding、国产算力适配 |
| **Qwen** | Qwen3-Max / 3.8-Max（1T→2.4T） | 混合线性注意力 + 超稀疏 MoE + MTP | Gated DeltaNet + Gated Attention 混合 | 开源生态最全、长文本效率最高 |

DeepSeek 注意力演进主线（本路线图阶段 4 的骨架）：
**MHA → MLA（V2/V3，低秩压缩 KV）→ DSA（V3.2，索引 + top-k 稀疏选择）→ CSA/HCA（V4，两级压缩 + 稀疏双管齐下）**

---

## 五、全部参考链接

**课程与代码**
- Karpathy 神经网络入门：<https://www.youtube.com/watch?v=VMj-3S1tku0>
- Let's build GPT from scratch：<https://www.youtube.com/watch?v=kCc8FmEb1nY>
- nanoGPT：<https://github.com/karpathy/nanoGPT>
- minbpe：<https://github.com/karpathy/minbpe>
- modded-nanoGPT：<https://github.com/KellerJordan/modded-nanoGPT>
- litgpt：<https://github.com/Lightning-AI/litgpt>
- 动手学深度学习（中文）：<https://zh.d2l.ai/>

**论文**
- RoPE：<https://arxiv.org/abs/2104.09864>
- Mixtral（MoE）：<https://arxiv.org/abs/2401.04088>
- DeepSeek-V2（MLA）：<https://arxiv.org/abs/2405.04434>
- DeepSeek-V3（FP8、MTP、auxiliary-loss-free）：<https://arxiv.org/abs/2412.19437>
- DeepSeek-V3.2（DSA）：<https://arxiv.org/abs/2509.01064>
- DeepSeek-V4 技术报告：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf>
- DSpark（推测解码）：<https://arxiv.org/abs/2607.05147>
- DPO：<https://arxiv.org/abs/2305.18290>

**开源权重与仓库**
- DeepSeek 开源主页：<https://github.com/deepseek-ai>
- DeepSeek-V4 模型合集：<https://huggingface.co/collections/deepseek-ai/deepseek-v4>

**技术解读（中文）**
- DeepSeek 注意力演进：从 MLA 到 DSA 再到 CSA/HCA：<https://zhuanlan.zhihu.com/p/2054620394934607925>
- DeepSeek-V4 技术报告深度解析：<https://cloud.tencent.com/developer/article/2661899>
- DeepSeek V4 深度解析（1.6T 开源 MoE）：<https://news.qiniu.com/archives/1777261349098>
- 十分钟读懂 DeepSeek-V3.2 稀疏注意力 DSA：<https://zhuanlan.zhihu.com/p/1959636888123049941>
- DSpark 技术要点拆解：<https://zhuanlan.zhihu.com/p/2054514797270538138>

---

*生成于 2026-08-29，基于当时公开的模型与技术信息整理。*
