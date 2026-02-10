# Text Mining | 03: Vector Semantics | 语义向量与词嵌入


## word2vec

### 用比喻来说明

给一大群词语安排座位，关系越密切的词语，互相离得就越近。你只有一本厚厚的书，记录了他们在不同场合的聊天记录。

#### 训练过程

安排者拿到了一个巨大的宴会厅（也就是向量空间），开始随机地给每个词语安排一个座位。

然后，他开始阅读那本聊天记录，一次只看一句话，比如：“程序员用键盘写代码。”以“键盘”为中心，看到了旁边的次“程序员”和“代码”。

“拉近”操作（正）：把“键盘”和“程序员”在座位图里的位置，拉近一点点；把“键盘”和“代码”的座位也拉近一点点。

“推远”操作（负）：随机抓几个词语，比如“猫”、“蛋糕”，安排者心想：“蛋糕”和“键盘”应该关系不密切，因为没有在聊天记录里一起出现过，于是他把“蛋糕”的座位从“键盘”那里推远一点点。

#### 不断重复以上操作

经过若干次调整后：程序员、键盘、代码……这些词聚集在了一起；医生、手术刀、病历……这些词也聚集在了一起；猫、猫粮、毛线球……这些词也聚集在了一起。

#### 回到Word2Vec本身

- 每个词的座位===代表了每个单词的**词嵌入向量（Word Embedding）**
    - 词嵌入向量：一个数字列表（也就是一个向量）
    - 通常包含几百个小数
    - 代表一个词在某个高维语义空间中的坐标
    - 最终计算机通过比较不同词汇的词嵌入向量的相似度，来判断不同词汇在意义上有多接近。
- 聊天记录===代表了用来训练的海量文本
- “拉近”和“推远”的操作===代表了训练算法（**Skip-gram with negative sampling**）
    - 把复杂的分类问题，转换成了二元判断的问题。

### 局限

Word2Vec无法处理多义词，比如，一个词语可能有两类不同的意义，于是就会在向量空间里处于两类词汇之间一个尴尬的中间位置。

## Preliminary knowledge

### Distributional hypothesis

（Linguistics）
Words that occur in similar contexts tend to be similar
在相似上下文中出现的词语也趋向于相似。

### Vector Space Model（VSM）

vector space：维数等于词汇表的大小。每一个单词都用向量来表示，基于单词之间的“同时出现次数”（co-occerrence）。

{{<admonition example "比如：">}}

digital 和 computer 同时出现较多，所以，表示 digital 的向量中，computer 那一位，数值就比较大；而比如 sugar 那一位，数值可能就非常小。

digital 向量可能长这样：[0, 0, …, (many zeors..), 999, 1789, 4,2,…] 其中 1789 那个就是 computer 和 digital 共同出现过这么多次。

{{< /admonition >}}

- 在词向量中，每一个单词，都是一个 discrete dimension。
- 向量有许多的 0，是稀疏的（sparse），因为对每个单词来说，绝大多数的单词都不会更它大量地同时出现。

---

其他的语义表示方法：topics（*topic modelling，Lecture 9 中讨论*）；稀疏语义向量 — 词嵌入（word embeddings）— 密集（dense）、让相似的单词在向量空间中距离得更近。

### Feed-forward neural networks

多层网络，没有循环。输入单元、隐藏单元、输出单元。全连接。

二元分类：只有一个输出节点，y 代表其中 positive output 的概率

多类别分类：每一个类别都有一个输出节点（分几类，就有几个输出节点），y 代表某个类别的概率。所以，输出层代表了概率分布。比如：[0.9, 0.03, 0.02, 0.05] 就代表一个分 4 类的分类问题。

如何得到概率分布：通过 Softmax 函数
$\text{softmax}(y_i)=\frac{e^{y_i}}{\sum_{j=1}^d e^{y_i}} \text{ for } i\leq i \leq d$
d = dimensionality（即：分成几类）
输入一个向量，输出一个概率分布向量，让向量中每个值的总和为 1。

## Word embeddings

词嵌入是稠密向量，在连续的稠密向量空间中。（continuous dense vector space）向量空间相对来说是低维的。（10000 → 100）

向量中的数值（也代表向量在向量空间中的位置）是隐式的（latent）。人无法理解它们的具体含义。

符合分布假说：语义和句法上相似的单词，会距离更近。

---

> （以下部分，是阅读 *J&M: Dan Jurafsky and James H. Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) (3rd ed), 2025* 第五章的阅读笔记与思考。）

- **Distributional hypothesis**（分布假设）：语言学的概念，意思是，出现在类似语境中的词，往往意义相近。关键是：词在分布上的相似性与意义上的相似性之间的联系。
- Embeddings（嵌入）：从文本中词的分布学习到的词的含义的向量化表示。

{{<admonition info "主要技术：">}}

Word Embedding 的生成：Word2vec （尤其是其中的 Skip-gram with negative sampling）

论文：

- Efficient Estimation of Word Representations in Vector Space: [[PDF]](https://arxiv.org/abs/1301.3781)

- Distributed Representations of Words and Phrases and their Compositionality: [[PDF]](https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html)

{{< /admonition >}}

### Vector Semantics 向量语义

思想：

1. 用多维空间中的一个点来表示词的情感意义，比如用三个维度分别表示愉悦度（valence）、唤醒度（arousal）、支配度（dominance）。（1957 Osgood等人）
2. 语言学中的分布思想：用词在语言中出现的上下文分布来定义其意义。一个词的含义由它出现在哪些环境、与哪些词相邻来决定。（1950 Joos、1954 Harris、1957 Firth）

**Vector semantics：**向量语义的核心思想，是将词语表示为多维语义空间中的一个点（该空间通过不同方式从单词临近词的分布中推导而来）。用于表示单词的向量，成为**嵌入**（**embeddings**）。

#### Simple count-basaed embeddings 简单的基于计数的词嵌入

这是一种最简单的词向量模型，基于共现矩阵（co-occurrence matrix）。我们定义一种特别的共现矩阵：词-上下文矩阵（word-context matrix）。

- 行：词表中的每个目标词（target）
- 列：词表中每个上下文词（context）
- 单元格：目标词与上下文词在附近（nearby）共同出现的次数

向量空间（vector space）：由向量构成的集合，其特征由维数决定（dimension）。比如：[1, 0, 0, 0] 这里的维度就是4。

这种方法，每个向量的维度等于词汇表的维度（几万）。

#### Cosine for measuring similarity余弦相似度衡量词语相似性

NLP（自然语言处理）中，最常用的相似度度量是：余弦相似度——基于线性代数中的点积（dot product），也成为内积（inner product）： $v\cdot w=\sum_{i=1}^N v_iw_i=v_1w_1+v_2w_2+\cdots+v_iw_i$

- 点积越大，通常说明：两个向量在相同维度上都有较大值
- 如果点积为0，则说明：两个向量垂直；两个向量非常不相似

##### 点积的缺陷 & 点积归一化

点积偏向长向量。向量的长度定义为： $|v|=\sqrt{\sum_{i=1}^N v_i^2}$  

如果一个向量比较长，则意味着它的每个维度都有比较大的值，而它与其他向量计算出来的点积也会更大。

前面说的基于count的向量语义方法，越频繁出现的词语，它的向量长度越大，因此，与任何词的点积都会比较大。但它并不应该与任何词都有较高的相似度——我们希望相似度的计算不受词语的出现频率的影响。

把原始的点积除以向量的长度，实现点积归一化（normalized dot product）： $\cos(\theta)=\frac{v\cdot w}{|v||w|}$。归一化的点积结果等于两个向量之间夹角的余弦值。两个向量v和w之间的余弦相似度的计算如下：

$$
\text{cosine}(v,w)=\frac{v\cdot w}{|v|~|w|}=\frac{\sum_{i=1}^Nv_iw_i}{\sqrt{\sum_{i=1}^Nv_i^2}\sqrt{\sum_{i=1}^Nw_i^2}}
$$

- 向量完全相同：余弦相似度=1
- 向量完全不相关（正交）：余弦相似度=0
- 向量完全相反：-1 （但这里不会出现，因为词计数非负）

### Word2vec

前面介绍的基于count的方法，词语被表示为稀疏（sparse）、且很长的向量，向量的维度等于词汇表的大小（有多少个不同的词，就有多少个维度）。所以这些向量维度巨大、大部分都是0（非常稀疏）。下面介绍的更强的方法：词嵌入（embeddings）。

#### 词嵌入 Embeddings

词嵌入是短小、密集的向量。

- 维度通常为50~1000
- 每个维度并没有清晰的含义
- 每个值都是实数，也可能为负数

NLP的任务中，比起上万的维度，使用词嵌入把词语表示为300维的向量，效果更好：

- 我们的分类器（classifier）需要学习的权重少了很多
- 更小的参数空间对于泛化（generalization）更有帮助，可以避免过拟合。
- 更好地找到同义词（synonyms）

接下来介绍一种计算词嵌入的方法：skip-gram with negative sampling（SGNS）。

#### Skip-gram with negative sampling (SGNS)

{{<admonition info "参考链接：">}}

1. [让电脑听懂人话：理解 NLP 重要技术 Word2Vec 的 Skip-Gram 模型](https://tengyuanchang.medium.com/%E8%AE%93%E9%9B%BB%E8%85%A6%E8%81%BD%E6%87%82%E4%BA%BA%E8%A9%B1-%E7%90%86%E8%A7%A3-nlp-%E9%87%8D%E8%A6%81%E6%8A%80%E8%A1%93-word2vec-%E7%9A%84-skip-gram-%E6%A8%A1%E5%9E%8B-73d0239ad698)

2. [Word2Vec Tutorial - The Skip-Gram Model](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

3. [Word2Vec Tutorial Part 2 - Negative Sampling](https://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)

{{< /admonition >}}

Word2vec模型中，主要有CBOW与Skip-gram两种模型。CBOW是给定上下文来预测输入的字词，而Skip-gram是给定输入字词后，来预测上下文。这里讨论Skip-gram模型。

我们的任务是：训练一个神经网络，在给定一段句子中的一个字词后，可以告诉我们其他字词出现在它附近的概率。

{{<admonition example "举例说明：">}}

`The quick brown fox jumps over the lazy dog.`

{{<image src="/images/course_note/tm-03-1.png">}}

把 window size 设为 2，图中蓝色部分的字词为输入的词

- 如 “the” 这个词可以产生（the，quick）和（the，brown）两对训练样本
- 而 “quick” 这个词可以产生（quick，the）、（quick，brown）和（quick，fox）这三对训练样本

根据输入的语料，我们训练的神经网络就会开始统计每一组成对出现的字词出现的次数。

比如，可能（cake，sweet）出现的次数会比（cake，AI）出现的次数多——意味着 cake 和 sweet 的含义更接近，sweet 出现在 cake 附近的概率，要高于 AI 出现在 cake 附近的概率。

{{< /admonition >}}

在神经网络中，输入层是一个用 one-hot 编码表示的向量（1 x 10000），而输出层也是一个相同维度、用 ont-hot 编码表示的向量，会转换成概率分布，用来表示每个字词出现在输入词附近的概率。

- 隐藏层（hidden layer）中，原论文使用了 300 个 features，让神经网络变成了一个 10000 行 x 300 列的矩阵。神经网络的目标，就是训练出隐藏层中，这个 10000 x 300 尺寸的矩阵中每一个值。
    - 举个例子：假设输入向量只用 5 维来表示一个词，隐藏层中只设定了 3 个features
        
        {{<image src="/images/course_note/tm-03-2.png">}}
        
        矩阵相乘的结果，其实就相当于隐藏层的权重矩阵中，对应 one-hot 编码中为 1 的那个维度的那一行的数值的向量——这个例子里这个 1 x 3 维度的向量，就成为输入的字词的词向量（word vector）。
        
    - 回到一开始那个 10000 x 300 的矩阵，我们为每个输入词都会得到一个 1 x 300 的向量——用来表示输入的那个词。
- 我们把一个 1 x 300 的某个输入词的向量，乘上另一个词的权重向量，得到的数字再进行 softmax 函数处理，这个值就会变成一个 0 ~ 1 之间的概率数值，代表第二个词出现在我们输入的那个词附近的概率。

提高模型训练效率的方案：

1. 将常见的单词组合（word pairs）或是词组（phrase）当做是单个字来处理。
2. 对高频率出现的字词进行抽样，来减少训练样本的数目。
    - 比如 “the” 出现频率非常高，会产生大量（the，XXX）这这种训练样本，样本数目可能远超我们学习 “the” 这个词向量所需要的训练样本数；而且，对于学习 “XXX” 这个词的语义，帮助可能也不大。
    - 解决方案：**抽样（subsampling）**
3. 对于优化目标，使用负采样（negative sampling）的方式，让每个训练模型，可以只更新一小部分的权重，来降低运算的负担。

负采样（negative sampling）：对于每个训练样本，只更新一部分的权重，而非整个神经网络的权重都被更新。

---

Skip-gram with negative sampling 学习到的向量是静态词嵌入（static embeddings），每个词只有一个固定的向量，不随上下文变化——这与后来的 BERT 不同（BERT 是动态词嵌入）。

word2vec 的革命性思想：不去数在某个词附近另个词出现的次数，而是训练一个分类器，去看另个词在某个词附近出现的概率有多大？但我们不关心分类结果，而是**把分类器学习到的权重当做词向量。**

而训练标签来自文本本身，不需要人工标注，这种方式叫做自监督学习（self-supervision）。大量现代 NPL 模型例如 BERT 也依赖这个思想。

##### The model

Skip-Gram 模型的目标就是学习用来表示词语的词向量。

思想：构建一个简单的神经网络

但最后我们并不会用这个网络来完成某个任务，我们真正想要的是这个网络的**隐藏层的权重** —— 那就是我们想要的**词向量**。

- 一个输入层
    - 输入的是：ont-hot encoded 向量，长度等于整个词汇表的大小 （比如 10000）。
        - 例如我们输入 “cat” 这个词，向量中除了代表 “cat” 的那个位置是 1以外，其他所有位置都是 0。
- 一个隐藏层
    - 节点数决定了我们最终学习到的词向量的维度（比如300）
    - 输入层到隐藏层之间有一个巨大的权重矩阵 W1
        - 矩阵的行数 = 词汇表的大小（比如10000）
        - 矩阵的列数 = 词嵌入的维度（比如300）
        - 所以这里的权重矩阵是 10000 x 300 大小的，而因为输入是 one-hot 编码，所以它乘以这个权重矩阵的结果，就等同于直接从矩阵中取出对应 one-hot 为 1 的那一行。
        - 所以，权重矩阵 W1 的每一行，就是每个词的词向量。
- 一个输出层
    - 输出的是：长度和输入向量一样的巨大的向量，长度也等于词汇表大小（10000）
    - 向量中每个值，代表每个单词是输入词的上下文的概率。
    - 使用 Softmax 函数，来确保这些值的总和等于 1，形成概率分布。

{{<admonition example "举例说明正向传播的过程：">}}

1. 将输入词 “cat” 的 one-hot 向量（比如第 10 位为 1）输入给网络
2. 这个 one-hot 向量乘以权重矩阵 W1（10000 x 300）。
    - 相当于：从 W1 中选择了第 10 行。这一行就是 “cat” 这个词的词嵌入向量（embedding）
3. 这个 300 维的向量（h）被传递给输出层
4. 为了得到输出层，用 h 乘以第二个权重矩阵 W2（300 x 10000）
5. 这个 1 x 10000 的结果向量，就是每个词的“概率得分”，我们对它应用 Softmax 函数，将其转换为概率分布
6. 最后结果是：网络输出一个 1 x 10000 的概率分布向量。向量中每个值都代表了词汇表中对应单词是 “cat” 这个输入词的上下文的概率。

{{< /admonition >}}

##### 训练过程

训练的目标：调整权重矩阵 W1 和 W2，使网络的预测结果更准确。

用反向传播和梯度下降。

{{<admonition example "用举例来解释训练的过程：">}}

当网络看到训练样本 （cats, animal）时：

1. 先进行正向传播，得到一个预测的概率分布
2. 计算这个预测与真实目标（即：一个在 “animal” 位置为 1，其他位置为 0 的 ont-hot 向量）之间的误差
3. 这个误差会被反向传播，用来更新权重矩阵 W1 和 W2

让输入 “cats” 时，输出里 “animal” 的概率变高。

“dogs” 同理。久而久之，它们的词向量（也就是 W1 中的对应两行）就会变得越来越相似。

{{< /admonition >}}

##### Negative sampling 负采样

之前我们的任务是让模型预测上下文词，我们把它变为：判断一个词是“好”是“坏”。

举例说明：模型会接收一对词，比如（cat，animal），然后输出一个概率，用来判断它们是不是一个“中心词-上下文词”的组合。如果是，我们希望网络输出的概率接近 1；如果不是，我们希望网络的输出接近 0。

##### 正例和负例的采样

为了训练这个新的网络，我们从训练文本中提取词对（word pair），并给它们打上标签 1 或 0（1 代表正例，即真实的上下文词对；0 代表负例）。

- 正例：生成方式和此前一样。使用一个滑动窗口扫过文本，所有在窗口内形成的“中心词-上下文词”对都是我们的正例。
- 负例：对于每一个正例，我们从词典中随机抽取 k 个词，来生成 k 个负例。
    
{{<admonition example "举例解释负例的生成：">}}
    
例如：对于正例（cat，animal），我们可能随机抽取到 “dog”, “information”, “kitchen” 等作为负样本。然后将会生成以下这些带标签的训练样本：
    
- cat - animal - 1
- cat - dog - 0
- cat - information - 0
- cat - kitchen - 0

{{< /admonition >}}
    

采样的方法：并非完全地随机采样，不然像 “a”、“the”、“of” 这类高频词被抽中概率会非常高，而它们并不能提供很有价值的信息。我们希望抽到更有意义的词。因此，用一种特殊的抽样方法。

- 抽中某个词 w 的概率 P(w) 是根据它的**词频**计算的，具体公式详见其他资源。

##### 最终的网络结构

最终，网络的输出层不再是之前那个 10000 个神经元的 Softmax 输出层了，而是**只有一个神经元，使用 Sigmoid 激活函数，目标是输出一个介于 0 和 1 之间的概率数值。**

- 对比：
    - 原始 Skip-gram的任务：预测哪个词在上下文中
    - Skip-gram with negative sampling：判断这对词是不是邻居
    - 输出层：前者是 10000 个神经元 + Softmax；后者是 1 个神经元 + Sigmoid
    - 计算成本：前者十分高昂，后者很便宜

