

Text-Similarity Method Implemented by Pytorch
https://github.com/pengshuang/Text-Similarity

### match_zoo

nltk.download('stopwords')
nltk.download('punkt')

$HOME/nltk_data
├── corpora
│   ├── stopwords
│   │   ├── README
│   │   └── ....
│   └── stopwords.zip
└── tokenizers
    └── punkt.zip

Processing text_left with chain_transform of Tokenize => Lowercase => PuncRemoval => StopRemoval => NgramLetter => WordHashing

W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory


## 数据集
Chinese medical dialogue data 中文医疗对话数据集 => cmdd
https://github.com/Toyhom/Chinese-medical-dialogue-data



+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.80.02    Driver Version: 450.80.02    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 00000000:02:00.0 Off |                  N/A |
| 19%   33C    P0    38W / 180W |      0MiB /  8116MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+



### 文本匹配
本文匹配, 或称语义相似匹配, 是NLP领域最基础的任务之一, 包括信息检索, 问答系统等问题都可以看作针对不同样本集的文本匹配应用.

- 文档相似度
### 短文本相似度度
### 文档相似度
### 问答匹配

[文本匹配方法 paper笔记](https://zhuanlan.zhihu.com/p/45089113)

#### Representation model
- Siamese Network
- Deep Structured Semantic model

#### Interaction model
基于交互的模型认为文本间的*匹配特征*对预测有很大帮助, 通过不同策略构建匹配特征, 与深度模型结合使用, 尽可能保留重要的句子间相似信息.
- Text Matching as Image Recognition
- Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement
- Bilateral Multi-Perspective Matching for Natural Language Sentences
- Lexical Decomposition and Composition