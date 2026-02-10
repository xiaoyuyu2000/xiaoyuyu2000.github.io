+++
date = '2025-10-27'
draft = false
title = '模型量化技术'
author = 'RR'
tags = ['模型量化']
categories = ['大模型技术']
summary = "*A Visual Guide to Quantization* 模型量化技术笔记"
+++

{{< admonition note "参考链接" >}}

原文：[Maarten Grootendorst: A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)

中文翻译版：[模型量化技术可视化指南](https://mp.weixin.qq.com/s/dgS-yRVpGe_w1uzbcVctXg)

{{< /admonition >}}

## 1. LLMs存在的问题

大语言模型（LLM）的参数数量能达到几十亿，其中包括权重参数（weights）、激活值（activations）等。我们的目标：用尽量高效的方式表达这巨量的参数，以减少存储参数所需要的空间。

### 参数数值（value）的表示方法

数值通常以浮点数形式来表示，带有正负号和小数点。

这些数值由 bits 组成，也就是二进制数字。这些 bits 可以被分成三个部分：

- 符号位（Sign）
- 指数部分（Exponent）
- 小数部分（也称为尾数）（Significand / Mantissa）

符号位、指数部分、小数部分结合在一起，就能根据一组特定的 bits 来计算出一个具体的数值了。

#### FP16

1 个符号位 + 5 个指数位 + 9 个小数位

{{< admonition example "举个例子：" >}}

某个FP16（16位浮点数），其二进制表示为：0 10000 1001001000

{{< figure src="https://substackcdn.com/image/fetch/$s_!Bn0Y!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4783fd02-a138-40c7-82c7-79dd05a179e4_1472x772.png" >}}

- 符号位：0，代表符号位是 $(-1)^0=1$，符号为正
- 指数部分：10000，代表 $2^{2^4-(2^{(5-1)}-1)=1}=2^1=2$
- 小数部分：1001001000，代表 $2^{-1}+2^{-4}+2^{-7}=0.5703125$

所以这个浮点数实际表示为：$1\times 2\times 1.5703125=3.140625$

{{< /admonition >}}

### 内存限制问题

**用来表示数值的位数（bit）越多：**

- **得到的数值精确度越高**
- 能表示的数值范围越大
- 占用空间也越大

动态范围（dynamic range）：指的是某个数值表示法所能表示的所有数值的区间。

精度（precision）：相邻两个数值之间的间隔。（相邻两数值离得越近，说明这个数值表示法的精度越高）

- 我们可以计算出存储一个特定数值所需要的设备内存量。一个字节（byte）等于8位（bits），所以可以位大多数浮点表示形式制定一个基本的计算公式：
    - $memory=\frac{nr\_bits}{8}\times nr\_params$
        - nr_bits：数值表示法用来表示数值的比特位数
        - nr_params：模型中的参数数量
    - 实际应用中，模型推理阶段所需的显存（VRAM）量还受别的因素影响（比如模型处理上下文的大小、模型架构设计）

{{< admonition example "举例说明：假设有一个 700 亿（70B）参数的模型" >}}



- 使用 FP32（32 位浮点数，常称为全精度）数值表示法：
    - $memory=\frac{32}{8}\times70B=280 GB$
- 使用 FP64：64/8 * 70B = 560GB
- 使用 FP16：16/8 * 70B = 140GB

由此可见，有必要尽可能减少用于表示模型参数数值的 bits 数量。然而，bits 数减少，精度会降低，从而导致力模型准确性下降。

{{< /admonition >}}

**目标：减少用于表示模型参数的 bits 数量，同时又不损害模型的准确性。**

——这就是模型量化技术的目的。

## 2. 模型量化技术简介

模型量化的核心：将模型参数的精度从较高的位宽（bit-widths）（比如32位浮点数）降低到较低的位宽（例如8位整数）。

减少比特位数时，会出现精度损失。

模型量化的主要目的就是减少表示原始参数所需的 bits 数量，同时尽可能保留原始参数的精度。（上文已经说过）

### 常用的数据类型

除了32-bit全精度（full-precisoin，FP32）之外，还有多种不同的数据类型。

#### FP16

16-bit 浮点数，为半精度浮点数，也称为 FP16。

{{< admonition example "举例说明，将 32-bit 转换成 16-bit（半精度，FP16）浮点数：" >}}

某个 FP16（16 位浮点数），其二进制表示为：0 10000 1001001000

{{< figure src="https://substackcdn.com/image/fetch/$s_!VLOD!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff4ac888a-02b9-4153-915a-e103a12c33a4_1460x892.png" >}}

FP16 的数值范围比 FP32 窄得多，且数值有一定的精度损失（3.14159274… → 3.140625）

{{< /admonition >}}

### BF16

为了保持与原始 FP32 相似的数值范围，引入了 bfloat 16这一数据类型（即 BF16，brain-float 16），相当于“截断版的 FP32”。

{{< admonition example "FP32 与 BF16：" >}}

某个 FP16（16 位浮点数），其二进制表示为：0 10000 1001001000

{{< figure src="https://substackcdn.com/image/fetch/$s_!_2X4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F172c93aa-58ae-4d11-8cb7-2917c265cb68_1460x936.png" >}}

表示的数值范围是相同的，但 BF16 只有 16 bits。BF16 在深度学习领域得到了广泛应用。

{{< /admonition >}}

#### INT8

进一步减少 bits 的数量，就要使用整数表示法了，比如仅有 8 bits 的 INT8。INT8 占用的 bits 数量仅仅为 FP32 全精度浮点数的四分之一。

{{< admonition example "FP32 与 INT8：" >}}

某个 FP16（16 位浮点数），其二进制表示为：0 10000 1001001000

{{< figure src="https://substackcdn.com/image/fetch/$s_!Wx2P!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffa37a58d-1f5a-433c-b235-5b073596bbca_1460x848.png" >}}

FP32 的数值范围： $[-3.4e^{38},~3.4e^{38}]$

INT8 的数值范围： $[-127,~128]$

{{< /admonition >}}

实际应用中，并不需要将 FP32 所表示的全部数值范围都映射到 INT8，我们只需要找到一种方法，将数据（即实际的模型的参数）范围映射到 INT8 即可。

常用的压缩（squeezing）和映射（mapping）方法包括对称量化、非对称量化，它们都是线性映射（linear mapping）的不同形式。

接下来讨论*将 FP32 量化成 INT8 的方法*。

### 对称量化（Symmetric Quantization）

对称量化中，原本浮点数的值域会被映射到量化空间（quantized space）中的一个以零为中心的**对称**区间。量化前后的值域都是围绕零点对称的。

{{< figure src="https://substackcdn.com/image/fetch/$s_!R1o0!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F730bbb8a-3a44-47f6-aefe-f652b117ae22_1124x600.png" >}}

在 FP32 中表示零的值，在 INT8 中依然正好表示零。

对称量化的一种经典方法：绝对最大值（absmax，absolute maximuum）量化。

#### **量化的过程**

1. 从一组数据中找出**最大的绝对值 α**，以此作为线性映射的范围（从 -α 到 +α）。
2. 计算**比例因子（s，scale factor）**：
    - $s=\frac{2^{b-1}-1}{\alpha}$
        - b 是我们想要的量化结果的 bits 数（这里 INT8 为 8）。
        - α 是最大的绝对值
3. **计算量化结果**：
    
    $X_{quantized}=round(s\cdot X)$

{{< admonition example "举例说明： $X=[5.47,~3.08,~-7.59,~0,~-1.95~,-4.57,~10.8]$" >}}

{{< image src="https://substackcdn.com/image/fetch/$s_!PeHR!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F782beaa8-340f-45b8-ba7f-20491f66867a_1172x848.png" >}}

- α = 10.8，所以线性映射范围设为 -10.8 ~ 10.8
- 比例因子： $s=\frac{2^{8-1}-1}{10.8}=\frac{127}{10.8}\approx 11.76$
- 量化： $X_{quantized}=round(11.76\cdot X)$
    $=round([64.3272, ~36.2208, ~-89.2584, ~0, ~-22.932, ~-53.7432, ~127.008])$
    $=[64, ~36, ~-89, ~0, ~-23, ~-54, ~127]$

{{< /admonition >}}

#### 反量化的过程

把量化后的 INT8 结果恢复成原始的 FP32 数值：使用之前计算出的比例因子 s ，对量化后的数值进行反量化（dequantize）。

**量化误差（quantization error）**：指的是原始值经过量化，然后再反量化回到原始数值表示法这个过程中，原始值（original values）与反量化值（dequantized values）之间的差值。

- 反量化的计算：
    - $X_{dequantized}=\frac{X_{quantized}}{s}$
    - 量化后的结果，除以比例因子 s，即可

{{< admonition example "继续以前边量化的例子来解释反量化：" >}}

- $X_{dequantized}=\frac{[64, ~36, ~-89, ~0, ~-23, ~-54, ~127]}{11.76}\approx[5.44, ~3.06, ~-7.54, ~0, ~, -1.96, ~-4.59, ~10.8, ~3.06, ~-1.96]$

整个量化与反量化的过程，如图：

{{< image src="https://substackcdn.com/image/fetch/$s_!EP4K!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5ea2d627-efc7-4a8a-9cf0-7a7020f1253d_1236x348.png" >}}

- 观察到，某些值量化到 INT8 之后，被分配到了相同的值（36），而反量化回到 FP32 时，它们也会变成相同的浮点数值。

量化误差：（比较原始值与反量化值）

{{< image src="https://substackcdn.com/image/fetch/$s_!QCY6!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe173b13b-ed99-4de0-a5e0-4b9114899b3f_1236x372.png" >}}

{{< /admonition >}}

一般来说，量化后的 bits 数越小，量化误差往往越大。

### 非对称量化（Asymmetric Quantization）

非对称量化并不是以零为中心对称的，它将浮点数范围中的最小值 β 和最大值 α 映射到量化范围（quantized range）的最小值和最大值。

我们在此讨论的方法称为零点量化。

#### 量化的过程

1. 找出数据中的最大值与最小值，以此来作为线性映射的范围。
2. **计算比例因子（s，scale factor）：**
    - $s=\frac{2^{b-1}-(-(2^{b-1}-1))}{\alpha-\beta}=\frac{128-(-127)}{\alpha-\beta}=\frac{255}{\alpha-\beta}$
3. 计算**零点（zeropoint）**：
    - $z=round(-s\cdot \beta)-2^{b-1}$
4. **计算量化结果：**
    - $X_{quantized}=round(s\cdot X+z)$

{{<admonition example "举个例子来解释非对称量化：">}}

{{<image "https://substackcdn.com/image/fetch/$s_!-PZj!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8ffa0c54-88bf-45c1-8636-bdb097bb8e6b_1172x848.png">}}

- 线性映射的范围为： $[-7.59,~10.8]$
    - 最小值与最大值到零点的距离是不相等的——“非对称”
- 计算比例因子 s： $s=\frac{255}{10.8-(-7.59)}\approx13.86$
- 计算零点 z： $z=round(-13.86\cdot (-7.59))-2^7=-23$
    - 即：FP32中的数值 0，在 INT8 中被映射到了 -23的位置。
- 计算量化结果：
    
    $X_{quantized}=round(13.86\cdot [...]+z)$
     $=[53, ~20, ~-128, ~-23, ~-50, ~-86, ~127]$

{{< /admonition >}}

#### 反量化的过程

- 反量化的计算：
    - $X_{dequantized}=\frac{X_{quantized}-z}{s}$

### 取值范围的映射与剪裁

将向量中的数值映射到更低的位表示形式，前边介绍的方法使得向量的整个范围都能被映射，但一个明显的缺点：有离群值（outlier）的时候，不太好处理。

**离群值（outlier）**：向量中的一个数值远大于（或远小于）其他所有数值，该数值就可以被视为离群值。

{{<admonition example "举个例子，来说明有离群值的时候，对数值的映射的危害：">}}

$X=[-0.32,~0.89,~0.45,~256]$  → 离群值：256

用非对称量化：

- 计算比例因子 s： $s=\frac{255}{256-(-0.32)}\approx0.99$
- 计算零点 z： $z=round(-0.99\cdot(-0.32))-2^7=-128$
- 计算量化结果： $X_{quantized}=round(0.99\cdot[-0.32, ~0.89, ~0.45, ~256])+z$
 $=[0, ~1, ~0, ~253]+(-128)$
 $[-128, ~-127, ~-128, ~125]$

所有较小的数值，都被映射到了相近的 INT8 数值，并因此失去了它们的独特特性。（为了“照顾到”离群值 256……）

{{< /admonition >}}

我们需要进行**剪裁（clipping）。**

一种简单的方法是：为原始值设定一个不同的动态范围，而所有离群值都会被映射到相同的值。

比如：假如我们将动态范围设置为 $[-5,~5]$，那所有超出这个范围的数值，都会被当做 -5 或 5 来看待，于是被映射成 -128 或 127。

- 优点：减少了非离群值的量化误差
- 缺点：增加了离群值的量化误差

### 校准过程（Calibration）

前文所说的手动选择一个动态范围，这个过程被称为校准（calibration）。

目的：找到一个能够包含尽可能多的数值（values）的范围，同时尽量减少量化误差。

不同的参数，校准方法不同。

#### 权重（和偏置项）Weights (and Biases)

LLMs中，可以将权重（weights）和偏置项（biases）视为预先确定的静态值，因为这些值在模型运行之前就确定了。偏置项的数量远小于权重的数量，偏置项通常被保留在更高的精度（比如 INT16），而量化的主要工作集中在权重的处理。

- 手动选择输入范围的百分位数
    - 会导致与前文提过的相似的裁剪（clipping）行为
- 优化原始权重和量化权重之间的均方误差（MSE）
- 最小化原始值和量化值之间的熵（KL 散度）

#### 激活值

LLMs 中，在整个推理过程中，持续更新的输入（input）通常被称为激活值（activations）。这些激活值往往需要通过激活函数进行处理，比如 sigmoid、 ReLU。

与权重不同，激活值会随着每次输入数据的改变而改变，因此很难进行精确量化。

校准权重和激活值的量化方法主要有两种：

- Post-Training Quantization（PTQ）：训练完成后进行量化
- Quantization Aware Training（QAT）：训练/微调过程中同时进行量化

## 3. 训练后量化（Post-Training Quantization，PTQ）

Post-Training Quantization（PTQ）是在训练完模型之后，再对模型的参数（包含权重、激活值）进行量化。

- 对于权重值的量化：（主要两种）
    - 对称量化（symmetric quantization）
    - 非对称量化（asymmetric quantization）
- 对于激活值的量化：因为我们不知道激活值的范围，所以需要通过模型的推理，来获取它们的 potential distribution（也就是在不同的输入与模型参数下，激活值可能出现的范围分布，然后我们根据这个分布来选择一个能包含大部分激活值的量化级别），然后再进行量化。（主要两种）
    - 动态量化（dynamic quantization）
    - 静态量化（static quantization）

### 动态量化（Dynamic Quantization）

当数据通过隐藏层时，其激活值会被收集起来。

随后，利用这些激活值的分布（distribution of activations），来计算量化输出值所需的零点（z）和比例因子（s）

- $s=\frac{255}{\alpha-\beta}$
- $z=round(-s\cdot \beta)-128$
- $X_{quantized}=round(s\cdot X+z)$

每一层都重复这个过程。每一层都有其独特的零点 z 和 比例因子 s，每一层的量化方案不同。

### 静态量化（Static Quantization）

静态量化在模型推理过程中，并不实时地计算零点（z）和比例因子（s），而是在模型训练或校准的过程中提前计算。

为了找到这些值，会使用一个校准数据集，并让模型处理这些数据，来收集可能的激活值分布（potential distribution）。

{{<image "https://substackcdn.com/image/fetch/$s_!05g6!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F46dd6825-2a1c-459e-88c0-022a01dcebf2_1194x636.png">}}

收集到这些数值后，就可以计算出必要的 s 和 z，以便于在推理过程中量化。

实际推理过程中，s 和 z 不需要重新计算，而是被应用于所有激活值，实现全局量化。

- 动态量化一般来说精度会更高，因为为每一个隐藏层都计算一次 s 和 z 值，但因此也会增加计算时间。
- 而静态量化虽然准确度略低，但由于已知了用于量化的 s 和 z，因此推理时更为高效。

### 探索 4-bit 量化的极限

两种在 HuggingFace 上常用的将量化位数进一步降到 6-bit、4-bit甚至 2-bit（但不建议低于 4-bit）的方法：

- GPTQ：全模型在 GPU 上运行
- GGUP：将一部分模型层从 GPU 转移到 CPU 上执行

#### GPTQ

采用非对称量化，逐层处理，每层都经过独立处理。

首先将模型的权重转换为 Hessian 矩阵（二阶偏导数矩阵，用于描述函数在其输入变量上的局部曲率。对于多变量函数，Hessian 矩阵可以用来了解函数在某点上的凹凸性、函数值对输入变量的变化有多敏感）的逆矩阵，它是模型损失函数的二阶导数，告诉我们模型输出对每个权重变化的敏感程度。

该过程展示了模型层中每个权重的重要性（权重的影响程度）。

与 Hessian 矩阵中较小值相关的全助攻更重要，因为它们的微小变化可能对模型性能产生重大影响。

#### GGUF

## 4. 训练中量化（Quantization Aware Training，QAT）

QAT 的目标是在训练过程中学习量化过程。

QAT 通常比 PTQ 更准确。

在训练过程中，引入所谓的“伪”量化，比如先将权重量化到例如 INT4 等形式，然后将它们再反量化回 FP32。

{{<image "https://substackcdn.com/image/fetch/$s_!ETJg!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc3a17734-65f8-45d7-8e4e-f7bc1c592577_1824x360.png">}}

这一过程使得模型在训练阶段进行损失值计算和权重更新时，能够考虑到量化误差。

QAT 试着探索损失函数中的“宽”最小值区域（“wide” minima），以尽可能减少量化误差，因为“窄”最小值区域（“narrow” minima）往往会导致更大的量化误差。

{{<image "https://substackcdn.com/image/fetch/$s_!GK9b!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa70ee37e-3b4f-4598-8eef-2a9ab13658c1_1200x640.png">}}

{{<image "https://substackcdn.com/image/fetch/$s_!2hF2!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb26d3f00-f599-4c75-beb4-21d87625b1d8_1200x640.png">}}

<!-- ### 1-bit LLM的时代：BitNet

### 权重的量化

### 激活值的量化

### 反量化过程（Dequantization）

### 所有 LLMs 实际上均为 1.58-bit

#### The Power of 0

#### 量化过程

## 总结 -->