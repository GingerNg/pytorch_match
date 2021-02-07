import numpy as np
from collections import Counter
import os
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
import codecs
import cfg
from utils import file_utils
import logging
from nlp_tools.tokenizers import WhitespaceTokenizer
import torch
from utils.model_utils import use_cuda, device
import pandas as pd
import random
import copy
# from sklearn import utils as sklearn_utils


def split_dataset(raw_path, train_path, dev_path, test_path):
    df = pd.read_csv(raw_path)
    df = df[df["category"] == "IM_内科"]  # 限定科室
    df = df[0:200]  # 小批量实验
    train_set, dev_test_set = train_test_split(df, shuffle=True, test_size=0.2)
    print(len(dev_test_set))
    print(len(train_set))
    dev_set, test_set = train_test_split(
        dev_test_set, shuffle=True, test_size=0.5)
    train_set.to_csv(train_path)
    dev_set.to_csv(dev_path)
    test_set.to_csv(test_path)
    print('split_dataset done')
    # df = sklearn_utils.shuffle(df)


def process(path):
    print(path)
    dataset = []
    df = pd.read_csv(path)
    data_list = df.values.tolist()
    for data in data_list:
        # print(data)
        subsequence = {
            "title": list(data[3]),
            "question": list(data[4]),
            "answer": list(data[5]),
            "category": data[6],
            "tag": 1}
        dataset.append(subsequence)
    for i in range(len(dataset)-1):  # negative sample
        subsequence = copy.deepcopy(dataset[i])
        # 随机采样
        inds = list(range(0, i)) + list(range(i+1, len(dataset)))
        ind = random.sample(inds, 1)[0]
        subsequence["answer"] = dataset[ind]["answer"]
        subsequence["tag"] = 0
        dataset.append(subsequence)
    random.shuffle(dataset)
    # print(dataset)
    return dataset


def process_data(config, train_path, dev_path):
    train_set = process(train_path)
    dev_set = process(dev_path)
    file_utils.write_json(config["train_set"], train_set)
    file_utils.write_json(config["dev_set"], dev_set)
    print("process_data　done")


def _batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))  # ceil 向上取整
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - \
            1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]  # ???

        yield docs  # 　返回一个batch的数据


class DatasetProcesser(object):
    def __init__(self, bert_path):
        super().__init__()
        self.tokenizer = WhitespaceTokenizer(bert_path)

    def get_examples(self, data, label_encoder):
        label2id = label_encoder.label2id
        examples = []
        for dat in data:
            # for text, label in zip(data['text'], data['label']):
            # label
            category_ids = label2id(dat["category"])
            ids = dat["tag"]
            dat["question"] = dat["question"][0:self.tokenizer.max_len-2]
            dat["answer"] = dat["answer"][0:self.tokenizer.max_len-2]
            question_token_ids = self.tokenizer.tokenize(dat["question"])
            answer_token_ids = self.tokenizer.tokenize(dat["answer"])
            examples.append(
                [ids, question_token_ids, answer_token_ids, category_ids])

        logging.info('Total %d docs.' % len(examples))
        return examples

    def data_iter(self, data, batch_size, shuffle=True, noise=1.0):
        batched_data = []
        if shuffle:
            np.random.shuffle(data)
            sorted_data = data
        else:
            sorted_data = data

        batch = list(_batch_slice(sorted_data, batch_size))
        batched_data.extend(batch)  # [[],[]]

        if shuffle:
            np.random.shuffle(batched_data)

        for batch in batched_data:
            yield batch

    def batch2tensor(self, batch_data):
        batch_size = len(batch_data)
        max_sent_len = 100
        # doc_labels = []
        # for doc_data in batch_data:
        #     # xul
        #     if len(doc_data[0]) >= max_sent_len:
        #         doc_labels.extend(doc_data[0][0:max_sent_len])
        #     else:
        #         doc_labels.extend(
        #             doc_data[0] + [0]*(max_sent_len-len(doc_data[0])))
        # # batch_labels = torch.LongTensor(doc_labels)

        token_type_ids = [0] * max_sent_len
        batch_inputs1 = torch.zeros(
            (batch_size, max_sent_len), dtype=torch.int64)
        batch_inputs2 = torch.zeros(
            (batch_size, max_sent_len), dtype=torch.int64)
        batch_inputs_type = torch.zeros(
            (batch_size, max_sent_len), dtype=torch.int64)
        batch_labels = torch.zeros(
            (batch_size, ), dtype=torch.float)

        for b in range(batch_size):
            question_token_ids = batch_data[b][1]
            answer_token_ids = batch_data[b][2]
            batch_labels[b] = batch_data[b][0]
            # ids = batch_data[b][0]
            for word_idx in range(max_sent_len):
                if word_idx < len(question_token_ids):
                    batch_inputs1[b, word_idx] = question_token_ids[word_idx]
                if word_idx < len(answer_token_ids):
                    batch_inputs2[b, word_idx] = answer_token_ids[word_idx]
                batch_inputs_type[b, word_idx] = token_type_ids[word_idx]

        if use_cuda:
            batch_inputs1 = batch_inputs1.to(device)
            batch_inputs2 = batch_inputs2.to(device)
            batch_inputs_type = batch_inputs_type.to(device)
            batch_labels = batch_labels.to(device)
        # print("batch_labels_shape:{}".format(batch_labels.shape))
        return (batch_inputs1, batch_inputs2, batch_inputs_type), batch_labels


class LabelEncoer():  # 标签编码
    def __init__(self):
        self.unk = 1
        self._id2label = []
        self.target_names = []
        # process label
        label2name = {0: 'Oncology_肿瘤科', 1: 'IM_内科', 2: 'OAGD_妇产科',
                      3: 'Surgical_外科', 4: 'Andriatria_男科', 5: 'Surgical_外科', }

        for label, name in label2name.items():
            self._id2label.append(label)
            self.target_names.append(name)

        def reverse(x): return dict(zip(x, range(len(x))))  # 词与id的映射
        self._label2id = reverse(self._id2label)

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    @property
    def label_size(self):
        return len(self.target_names)

    def label2name(self, xs):
        if isinstance(xs, list):
            return [self.target_names[x] for x in xs]
        return self.target_names[xs]
