from nlp_tools.tokenizers import WhitespaceTokenizer
import torch
from torch import nn
import logging
from transformers import BertModel
from utils.model_utils import use_cuda, device
import numpy as np
# from cfg import bert_path
import torch.nn.functional as F
import logging

# build word encoder
dropout = 0.15


class BertSoftmaxModel(nn.Module):
    def __init__(self, bert_path, label_encoder):
        super(BertSoftmaxModel, self).__init__()
        self.all_parameters = {}
        parameters = []
        self.dropout = nn.Dropout(dropout)

        self.tokenizer = WhitespaceTokenizer(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        # bert_parameters = self.get_bert_parameters()

        self.siam_dense1 = nn.Sequential(nn.Linear(768, 1024, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(1024, 1024, bias=True),
                                         nn.ReLU())
        self.siam_dense2 = nn.Sequential(nn.Linear(768, 1024, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(1024, 1024, bias=True),
                                         nn.ReLU())

        self.fc1 = nn.Sequential(nn.Linear(1024*2, 256*2, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(256*2, 64, bias=True),
                                 nn.ReLU(),
                                 )
        self.fc2 = nn.Linear(64, 1, bias=True)

        parameters.extend(
            list(filter(lambda p: p.requires_grad, self.siam_dense1.parameters())))
        parameters.extend(
            list(filter(lambda p: p.requires_grad, self.siam_dense2.parameters())))
        parameters.extend(
            list(filter(lambda p: p.requires_grad, self.fc1.parameters())))
        parameters.extend(
            list(filter(lambda p: p.requires_grad, self.fc2.parameters())))

        if use_cuda:
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters
        # self.all_parameters["bert_parameters"] = bert_parameters  # bert参数不训练
        self.pooled = False
        logging.info('Build Bert encoder with pooled {}.'.format(self.pooled))

    def encode(self, tokens):
        tokens = self.tokenizer.tokenize(tokens)
        return tokens

    def get_bert_parameters(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return optimizer_parameters

    def forward(self, batch_inputs):
        input_ids1, inputs_ids2, token_type_ids = batch_inputs

        sequence_output1, pooled_output1 = self.bert(
            input_ids=input_ids1, token_type_ids=token_type_ids)
        sequence_output2, pooled_output2 = self.bert(
            input_ids=inputs_ids2, token_type_ids=token_type_ids)
        # print(pooled_output1)
        # if self.training:
        #     sequence_output = self.dropout(sequence_output)
        logging.info("sequence_output_shape:{}, pooled_output_shape:{}".format(
            sequence_output1.shape, pooled_output1.shape))
        out1 = self.siam_dense1(pooled_output1)
        out2 = self.siam_dense2(pooled_output2)

        # out = torch.cat([out1, out2], dim=1)
        # score = self.fc2(self.fc1(out)).squeeze(dim=1)
        # score = torch.sigmoid(score)

        input1 = out1
        input2 = out2
        score = torch.cosine_similarity(input1, input2, dim=1)

        # labels = torch.(score)
        # print("score_shape:{}".format(score.shape))
        # score = score.view(score.shape[0] * score.shape[1], score.shape[2])
        return score
