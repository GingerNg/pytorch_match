

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
