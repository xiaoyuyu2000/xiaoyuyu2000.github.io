# Vit 模型：《An Image is Worth 16x16 Words: Transformers for Image Recognitoin at Scale》论文笔记


{{< admonition info "原始论文" >}}

**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (Dosovitskiy et al., 2021): [[PDF]](https://arxiv.org/abs/2010.11929)

- 一张图片就相当于 16x16 的（若干）单词——用于大规模图形识别的 Transformer

{{< /admonition >}}

## 简单介绍

这篇文章主要任务，就是通过 Transformer 结构和自注意力机制，把计算机视觉的任务，当做自然语言处理的任务去完成。主要思想就是把图片分成若干个 16x16 尺寸的 patches，把每个 patch 都当成 token，然后用监督学习训练图片分类的任务。

### Motivation

- Transformer 适用于 NLP（natural language processing，自然语言处理）任务
- 而 CNN（convolutional neural network，卷积神经网络）依然是计算机视觉（computer visoin，CV）领域的主流。
    - 于是很多工作在尝试将 Transformer 的自注意力机制用于 CV 领域。

## Vision Transformer (ViT) 的主要思想

{{< admonition quote "Chapter 1" >}}

Inspired by the Transformer scaling successes in NLP, we experiment with **applying a standard Transformer directly to images**, with the fewest possible modifications. To do so, we **split an image into patches** and **provide the sequence of linear embeddings of these patches as an input to a Transformer**. Image patches are treated the same way as *tokens (words) in an NLP application*. We train the model on *image classification* in *supervised* fashion.

{{< /admonition >}}

把 Transformer 直接用于图片：首先把图片分割成 patches，然后把这些 patches的线性嵌入（linear embeddings）的序列作为 Transformer 结构的输入。

图片的 patches 就跟 NLP 任务里的词元（tokens）（也就是单词）类似：每个 patch 就相当于 NLP 里的一个单词。

我们用监督学习的方式训练模型用于图片分类的任务。

- 序列长度的变化：（序列长度等于 patch 的个数）
    - 一个图片原本的尺寸：224x224=50124 个像素
    - 把它分成若干个 16x16 尺寸的 patches 之后：
        - 每一行和每一列的“序列数目”就变成了：224/16=14
        - 所以，序列长度就从原先的 50124 变成了 14x14=196
        - （就相当于 NLP 中，一个 196 个 token 的语句）

## ViT的结构

尽可能地保持原始的 Transformer 结构不更改——这是 ViT 的一项优势。

结构图如下：（来自论文 Figure 1）

{{< figure src="https://paper-assets.alphaxiv.org/figures/2010.11929v2/img-0.jpeg" caption="图1：ViT 模型架构 (Dosovitskiy et al., 2021)" >}}

### ViT结构的详细解释

1. 首先，给定一张图，把这张图分解成了若干个 patch
2. Linear Projection of Flattened Patches（线性投射层）：

    a. 把这些 patch 当成一个序列，每个 patch 都会经过一个线性投射层的操作，得到一个特征，也就是 Patch + Position Embedding
    - 相当于 NLP 中的原始语义+位置编码
    - 得到的结果就相当于 NLP 中的 token

    b. 利用 BERT 的 Extra learnable embedding 的特殊字符 cls 分类字符：
    - 也就是结构图中的星号部分的“token”，位置编码永远是 0
    - 我们只需要根据它的输出，作最后图片分类的判断。
3. Transformer Encoder：上一步得到的结果作为 Transformer 编码器的输入
4. MLP Head：通用的分类头
5. 分类：交叉熵函数

### 图片尺寸与序列大小

原先：224x224x3

每个 16x16x3 的 patch：16x16x3=768 个像素大小

总共分成了多少个 patch：224/16=14，14x14=196 个 patch （等于输入的序列大小）

——原先的图片被分成了 196 个 patch，每个 patch 的维度是 768

（NLP 任务中：一句话包含了若干个 token，每个 token 的维度是 512）

{{< admonition note "矩阵尺寸" >}}

- 输入：X → 196x768
- 经过线性投射层之后：197x768（加上了那个 Extra learnable embedding 的特殊字符 cls 分类字符，也就是加了 1）

多头自注意力：

- 总共 12 个头
    - 768/64=12
    - 而 NLP 中：是 512/64=8 个头

{{< /admonition >}}

