import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(current_path)
sys.path.append(current_path)
os.chdir("..")
# from Basic.tokenizers import WhitespaceTokenizer
from cfg import bert_path
from transformers import BertModel, BertTokenizer, BertForNextSentencePrediction, BertForMaskedLM, BertConfig
import torch
import numpy as np
# from nlp_tools.tokenizers import WhitespaceTokenizer

def run(mtd):
    if mtd == 1:
        samples = ["窝个三我"]
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        # 将句子分割成一个个token，即一个个汉字和分隔符
        tokenizer_text = [tokenizer.tokenize(i) for i in samples]
        # [['[CLS]', '中', '国', '的', '首', '都', '是', '哪', '里', '？', '[SEP]', '北', '京', '是', '[MASK]', '国', '的', '首', '都', '。', '[SEP]']]
        # print(tokenizer_text)

        input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenizer_text]
        print(input_ids)

        tokenizer = WhitespaceTokenizer(bert_path)
        # sent_words = ["34", "1519", "4893", "43"]
        sent_words = ["窝", "个", "三", "我"]

        token_ids = tokenizer.tokenize(sent_words)
        print(token_ids)

        bert = BertModel.from_pretrained(bert_path)
        sent_len = len(token_ids)
        token_type_ids = [0] * sent_len

        batch_size = 16
        max_doc_len = 8
        max_sent_len = 64
        batch_inputs1 = torch.zeros(
            (batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_inputs2 = torch.zeros(
            (batch_size, max_doc_len, max_sent_len), dtype=torch.int64)

        for word_idx in range(len(token_ids)):
            batch_inputs1[0, 0, word_idx] = token_ids[word_idx]
            batch_inputs2[0, 0, word_idx] = token_type_ids[word_idx]

        batch_inputs1 = batch_inputs1.view(
            batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_inputs2 = batch_inputs2.view(
            batch_size * max_doc_len, max_sent_len)

        # torch.Size([128, 64])   128 = batch_size*max_doc_len
        print(batch_inputs1.shape, token_ids)
        # input_ids = torch.tensor(token_ids)
        # token_type_ids = torch.tensor(token_type_ids)

        # token_type_ids：区分两个句子的编码（上句全为0，下句全为1）
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        sequence_output, pooled_output = bert(
            input_ids=batch_inputs1, token_type_ids=batch_inputs2)
        print(sequence_output.shape)  # torch.Size([128, 64, 768])
        print(pooled_output.shape)   #



if __name__ == "__main__":
    run(mtd=1)
