+++
date = '2025-10-24'
draft = false
title = 'Transformers 模型：《Attention is All You Need》论文笔记'
author = 'RR'
tags = ['Transformers', '论文', 'LLM']
categories = ['论文']
summary = "Transformers 模型：《Attention is All You Need》论文笔记"
+++

{{< admonition type=info title="参考链接" open=true >}}

1. **Attention is All You Need** (Vaswani et al., 2017)：[[PDF]](https://arxiv.org/abs/1706.03762)

2. 一位B站博主的视频讲解：[《Attention is all you need》论文解读及Transformer架构详细介绍](https://www.bilibili.com/video/BV1xoJwzDESD/?spm_id_from=333.1387.favlist.content.click&vd_source=63cc2e6b0d3af0b51a5072cad8f5af99)

{{< /admonition >}}

- 序列转导模型（sequence transduction models）：比如翻译任务，输入内容的前后顺序很重要。
- Transformer之前的模型：
    - RNN（recurrent neural network，循环神经网络）和CNN（convolutional neural network，卷积神经网络）
    - 包括编码器（encoder）和解码器（decoder）
    - 使用注意力机制（attention mechanism）增强
- Transformer结构的创新：
    - 完全摒弃了RNN和CNN（但仍然使用encoder-decoder结构）
    - 完全基于注意力机制

## 背景知识

### Feedforward Neural Network (FNN) 前馈神经网络

不适合序列转导任务。

序列转导任务：分词（Tokenization）→ 词向量表示（Embedding）：把词元用一组N维向量来表示，例如 $[0.2,0.4,-0.4,0.5]$ 这种 → 合并词向量（平均 or 拼接）

- 平均：完全丢失了词语的顺序

- 拼接：FNN需要固定维度的输入，对不同长度的句子处理效率低下；FNN将句子作为一个整体来处理，无法理解真正的先后顺序的关系。

### Recurrent Neural Network (RNN) 循环神经网络

RNN能解决的问题：

- 构建词序：RNN按时间顺序（token顺序）逐个处理输入
- 构建上下文依赖：RNN逐个地喂入词语，并有“记忆”机制
- 支持不定长输入：不再需要FNN那种固定长度的输入，句子多长都可以。

{{< admonition example "RNN 模拟人说话的过程示例" >}}

RNN像是在模拟人说话的过程，举个例子说明。（“我爱小猫”）

$$h_t=g(Wx_t+Uh_{t-1})$$    $$y_t=g(Vh_t)$$

**t1时刻**：
1. 输入第一个 token “我”
2. $x_1$ 与矩阵权重 $W$ 相乘，再加上一个初始化的隐藏状态 $h_0$ 乘上权重矩阵 $U$
3. 相加的结果输入给激活函数 $g(x)$，得到一个 $h_1$（$h_1$ 是用于下一时刻的输出）
4. $h_1$ 与另一个权重矩阵 $V$ 相乘
5. 相乘结果也经过一个激活函数 $g(x)$，得到 $y_1$（$y_1$ 是 $t_1$ 时刻的实际输出）

{{< /admonition >}}

### Encoder and Decoder 编码器 - 解码器结构

编码器处理输入，得到一个最终的编码结果；解码器使用这个编码结果，去给出最终的输出。

- **编码器**：其实就是一个RNN神经网络去掉了输出 $y$ 的部分，只保留输出 $h$ 的部分。
- 编码器将得到一个**上下文向量 $C$（Context vector）**
    - 上下文向量是对整个输入序列的语义编码，是一个固定长度的向量，涵盖了整个输入文本的语义信息。
    - $C$ 等于最后一个时间步的隐藏状态输出 $h_t$
    - $C$ 作为 Decoder 的输入，用于生成目标序列。
- **解码器**的解码方式：
    - 最简单的方式：第一个时刻，隐藏状态的输入 $s_0$，得到第一个时间步的输出结果（也就是第一个 token 的解码结果 “I”），这个输出结果会送入第二个时间步成为它的输入，而第一个时间步还得到一个隐藏状态的输出结果 $s_1$，$s_1$ 也会送入第二个时间步成为它的输入。进行下去直到得到所有时间步的输出结果。
    - 另一种方式：由于可能解码到后边的时候，上下文信息已经被稀释了很多，可以给每个时间步重新输入一次上下文向量 $C$。

#### Encode-Decode的问题

- 处理长序列时，有“遗忘”的问题：远距离依赖信息在传递过程中，会被稀释。
- 不同时间步的输入对当前时刻输出的“重要性”问题：所有时间步的输入，在计算当前时刻输出时，被等同对待，忽略了不同时间步对当前时刻输出的重要性可能存在差异。

### Attention Mechanism 注意力机制

在解码器的部分，给每个时间步重新输入上下文向量C的时候：给每一个时间步输入的上下文向量C的值不一样。

$$C_i=\sum_{j=1}^n\alpha_{ij}h_j$$

{{< admonition example "举例说明" >}}

比如之前那个例子：“我爱小猫” → “I love little cats”

$C_0=0.6h_1+0.1h_2+0.2h_3+0.1h_4$

$C_1=0.2h_1+0.7h_2+0.1h_3+0h_4$

$C_2=0.1h_1+0.1h_2+0.4h_3+0.4h_4$  → 更“注意”一下“小”旁边的“猫”字

$C_3=。。。$

{{< /admonition >}}

- 依然存在的问题：串行化计算
    - 递归计算、顺序计算——总要在前一个时间步结束后，才能进行下一个时间步的计算，导致训练过程无法并行化。
    - Transformer结构
        - 摒弃了RNN，为了解决顺序计算问题
        - 也不使用CNN（CNN可以解决顺序计算的问题，但却又引入了RNN已经解决掉了的远距离依赖随着序列变长而被稀释的问题）

## Transformer

### Transformer的结构

{{< figure src="https://paper-assets.alphaxiv.org/figures/1706.03762v7/ModalNet-21.png" caption="图1：Transformer 模型架构 (Vaswani et al., 2017)" alt="RNN Architecture" >}}

下文依然以“我爱小猫”翻译成“I love little cats”的翻译为例子来解释。

#### Part 1: Encoder 

1. Input Embedding：首先，经过词嵌入，将每个词都转换成机器能理解的向量形式
    - “我爱小猫”这四个字，每一个字都被转换成一个512维的向量
    - 词嵌入的向量包含词本身的**原始语义信息**
2. **Positional Encoding：位置编码**，通过一个公式，给每一个词都生成一个 512 维的位置编码的向量（公式详见 Section 3.5）。生成好的位置编码，直接通过相加的方式，叠加到每个词的词嵌入向量上。
    - $PE_{pos,2i}=sin(pos/10000^{2i/d_{model}})$
    - $PE_{pos,2i+1}=cos(pos/10000^{2i/d_{model}})$
    - 为什么位置编码如此复杂（还涉及到傅里叶计算）？不能直接给每个词一个位置编号0, 1, 2…？
        - 其实有的模型（比如 BERT）就是这样操作的
        - 但这种编码是标量，信息量太少，模型无法理解，比如“第3个词和第10个词的关系有多远”，尤其是注意力机制是依靠向量计算的（比如点积），而单个数字在向量空间里几乎没法建立复杂的关系。
        - 所以要把位置信息变成一个高维向量，其中包含的信息就可以很丰富了，比如：能让模型感受到不同位置之间的差异、捕捉到“某两个词之间的位置差”等等相对关系。
3. **Multi-head Attention（多头自注意力）：**
    - 涉及到 $Q$、$K$、$V$ 矩阵。最后会使矩阵在包含原始的语义信息、上一步增加的位置编码之后，又增添了**上下文信息**。

{{< admonition note "单头自注意力机制的解释" >}}

QKV：Q（query）、K（key）、V（value）

- 首先，输入的是一个 4*512 的矩阵，代表“我爱小猫”。
- 然后，和三个矩阵，WQ、WK、WV 进行矩阵乘法。这三个矩阵都是 512*512 的尺寸，里面存储的是权重。
    - 得到三个 4*512 的矩阵作为结果，分别是 Q、K、V
- 将 Q、K、V 三个矩阵代入 Attention 计算公式（Section 3.2.1，公式 1）：
    
    $$
    \text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
    $$
    
    - 得到一个 4*512 的矩阵作为输出。

**多头自注意力机制：**

- WQ、WK、WV 这三个权重矩阵，大小为 512*64
- 于是得到的 Q、K、V 矩阵大小为 4*64
- 多头：即 WQ、WK、WV 一共有 8 组（512/64=8）。
    - 而得到的 Q、K、V 矩阵也一共有8组。
- 将这 8 组 4*64 的矩阵拼接起来：
    - 最后得到三个 4*512 大小的矩阵（和单头结果是一样大小的）
    - 再经过一个线性层
    - 得到输出，4*512

{{< /admonition >}}

4. Add & Norm （残差连接 & 归一化）
    - 残差连接：把经过多头注意力机制处理前那一步的数据，直接相加到经过处理后的数据上——避免如果多头注意力处理得很差时，可能导致模型很差
    - 归一化：使数值趋于稳定
5. Feed Forward（前馈神经网络）
6. Add & Norm （又一次，残差连接 & 归一化）

到这一步位置，就得到了编码器最终的输出结果。矩阵维度是没有变化的（这个例子里是 4*512）。

#### Part 2: Decoder

