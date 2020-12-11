

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