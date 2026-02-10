# Text Mining | 01: Introduction


Text mining: Automatic extraction of knowledge from text
从文本中自动地提取知识（信息）

Text: unstructured; Knowledge: structured

Big text data —(Text retrievl)→ Small relevant data —(Text mining)→ Knowledge → Many applications

**Challenges of text data:**

1. unstructured
2. noisy (source, content, labels)
3. language is infinite
    - **Heap’s Law**: 语料库（corpus）很小的时候，新词的数目增长非常快，而且会无限地继续增长。
4. language is ambiguous

## Text mining pipeline

1. Information Retrival (IR)
2. Pre-processing
3. NLP

# Types of text processing tasks

## Text classification/clustering

为每个 document 分类，document 可以是任何文本类型（报纸文章、推文、email、文字信息、一个句子。类别可以是任何的 label（主题、相关度、作者、情感等）。

## Sequence labelling (= named entity recognition)

为文本中的**每个词**分类。

例如：在文本中识别人名、地名（NER）。

## Text-to-text generation

输入文本，输出文本。

- 总结（summarization）：sequence-to-sequence
- 翻译

# Text mining models

ML 的范式：

1. supervised learning
    1. 训练 featured-based 模型
    2. lightweight，explainable
2. transfer learning
    1. 根据任务，fine-tune 预训练模型（如 BERT）
    2. 有足够的 labelled data 的 TM 任务的最佳选择
3. in-context learning
    1. 给 LLM prompt 和一些例子作为引导
    2. 没有 labelled data 的 TM 任务的一个选择

**监督学习和迁移学习都需要大量的 labelled data。**

## 文本作为分类对象 & 文本作为序列（sequence）

- 把文本作为整体进行分类时，文字顺序（word order）没有那么重要，而把文本作为序列时，文字顺序比较重要。
    - 文本作为分类对象：垃圾邮件处理、新闻分类
    - 关注整体内容
    - 文本作为序列：给句子里每个词打标签（人名、地名、动词等）
    - 关注上下文结构、词序

## 文本分类

- 基于特征的文本分类方法把文本表示为“bag of words”。
    - bag-of-words 模型里，集合里的每个单词都是一个特征（feature）。
    - 传统的 bag-of-words 模型中，词序、标点符号（punctuation）、句子和段落的分界线都不重要。
    - 词袋模型的缺点：
        - 维度爆炸 - 维度太高了，假如词典里有 10000 个词，每个单词的向量长度就都是 10000。
        - 稀疏（sparsity） - 词向量的大部分位置都是 0，浪费资源

### Zipf’s Law

Frequency 与 rank 成反比：一个词出现的频率，与它在频率表里的排名成反比。也就是说，在频率表里越靠前的词，越频繁出现；很靠后的词，几乎不出现。

- Lont-tail distribution：头部（head）是极少数出现的词，但出现次数最多；长长的尾巴（tail）是绝大多数的词，但它们包含各种含义却出现次数非常少。
- 所以：要去掉“头部”这些词，也就是停用词（stop word）；也要小心处理尾部，因为数据太少。

### 文本的稠密表示

word embeddings（词嵌入）：低维、稠密、隐式的（latent）、语义向量。

比如：word2vec、GloVe、BERT

*在 Lecture 3 讨论词嵌入。*

## 文本序列

从文本中提取知识，就要关注词序、标点符号、大小写。

识别人名、时间、电影名字等等。

# Transformer models

sequence-to-sequence 任务：输入文本，输出文本。

比如：翻译。

Transformer 结构有两个模块：Encoders，Decoders。分别负责处理输入文本和生成输出文本。

*在 Lecture 6 讨论 Transformer。*

### BERT

Bidirectional Encoder Representations from Transformers (BERT)：只有 **encoder**；输入文本，输出 embeddings。

适用于：分类（例如 sentiment）、序列标注（sequence labelling，例如 NER）。

### GPT

Generative Pre-trained Transforer (GPT)：只有 **decoder**；输入 prompt，输出文本。

decoder 模型巨大的时候，比如 GPT-4，也可以做 sequence-to-sequence 任务、甚至是分类任务，只要在 prompt 里很好地描述。

### Encoder-decoder models

T5：总结、翻译

*在 Lecture 6、7 讨论 LLMs。*
