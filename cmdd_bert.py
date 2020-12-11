from corpus_preprocess import cmdd_preprocess as preprocess
import cfg
from models.text_bert import BertSoftmaxModel
from models.optimizers import Optimizer
import torch.nn as nn
import torch
import numpy as np
from utils import file_utils, model_utils
from evaluation_index import scores
import os
import time
from models import loss_factory
from utils.constants import MthStep


config = cfg.config
# dataset_name = "demo"
dataset_name = "cmdd"
model_name = "bert_match"

config.define("raw_path", "data/raw_data/%s" % "Chinese-medical-dialogue-data-master", "path to raw dataset")
config.define("save_path", "data/dataset/%s" % dataset_name, "path to save dataset")
config.define("glove_name", "840B", "glove embedding name")
# glove embedding path
# glove_path = '/data/dh/glove/glove.840B.300d.txt'
glove_path = './data/glove/glove.840B.300d.txt'
#glove_path = os.path.join(os.path.expanduser(''), "utilities", "embeddings", "glove.{}.{}d.txt")
config.define("glove_path", glove_path, "glove embedding path")
config.define("max_vocab_size", 50000, "maximal vocabulary size")
config.define("max_sequence_len", 200, "maximal sequence length allowed")  # 最大序列长度
config.define("min_word_count", 1, "minimal word count in word vocabulary")
config.define("min_char_count", 10, "minimal character count in char vocabulary")

# dataset for train, validate and test
config.define("vocab", "data/dataset/%s/pd_vocab.json" % dataset_name, "path to the word and tag vocabularies")

config.define("train_set", "data/dataset/%s/pd_train.json" % dataset_name, "path to the training datasets")
config.define("dev_set", "data/dataset/%s/pd_dev.json" % dataset_name, "path to the development datasets")

config.define("dev_text", "data/raw/LREC/2014_dev.txt", "path to the development text")

config.define("test_set", "data/dataset/demo/bert_ref.json", "path to the ref test datasets")
config.define("test_text", "data/raw/LREC/2014_test.txt", "path to the ref text")
config.define("pretrained_emb", "data/dataset/demo/glove_emb.npz", "pretrained embeddings")


config.define("cell_type", "lstm", "RNN cell for encoder and decoder: [lstm | gru], default: lstm")
config.define("num_layers", 4, "number of rnn layers")
config.define("use_pretrained", False, "use pretrained word embedding")
config.define("tuning_emb", False, "tune pretrained word embedding while training")
config.define("emb_dim", 300, "embedding dimension for encoder and decoder input words/tokens")

config.define("train_batch_size", 16, "train_batch_size")
config.define("test_batch_size", 16, "test_batch_size")
config.define("epochs", 100, "epochs")
config.define("clip", 5.0, "clip")

label_encoder = preprocess.LabelEncoer()
dataset_processer = preprocess.DatasetProcesser(cfg.bert_path)

raw_path = os.path.join(cfg.proj_path, config["raw_path"], "all_data.csv")
train_path = os.path.join(cfg.proj_path, config["raw_path"], "train.csv")
dev_path = os.path.join(cfg.proj_path, config["raw_path"], "dev.csv")
test_path = os.path.join(cfg.proj_path, config["raw_path"], "test.csv")



def run(mtd="fold_split"):
    if mtd in [MthStep.train.value]:
    # build the network model
        if not cfg.RESUME_EPOCH:
            model = BertSoftmaxModel(cfg.bert_path, label_encoder)
        else:
            save_folder=os.path.join(cfg.proj_path, "data/bert_nn")
            print(' ******* Resume training from --  epoch {} *********'.format(cfg.RESUME_EPOCH))
            model = model_utils.load_checkpoint(os.path.join(save_folder, 'epoch_{}.pth'.format(cfg.RESUME_EPOCH)))

    def _eval(data):
        model.eval()  # 不启用 BatchNormalization 和 Dropout
        # data = dev_data
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_data in dataset_processer.data_iter(data, config['test_batch_size'], shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = dataset_processer.batch2tensor(batch_data)
                batch_outputs = model(batch_inputs)
                y_pred.extend(torch.max(batch_outputs, dim=1)
                              [1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

            score, dev_f1 = scores.get_score(y_true, y_pred)
        return score, dev_f1

    if mtd == MthStep.fold_split.value:
        preprocess.split_dataset(raw_path, train_path, dev_path, test_path)
    elif mtd == MthStep.process_data.value:
        preprocess.process_data(config, train_path, dev_path)
    elif mtd == MthStep.train.value:
        Train_data = file_utils.read_json(config["train_set"])
        Dev_data = file_utils.read_json(config["dev_set"])
        # 生成模型可处理的格式
        train_data = dataset_processer.get_examples(Train_data, label_encoder)
        dev_data = dataset_processer.get_examples(Dev_data, label_encoder)
        del Train_data, Dev_data
        # 一个epoch的batch个数
        batch_num = int(np.ceil(len(train_data) / float(config["train_batch_size"])))
        print("batch_num:{}".format(batch_num))
        # model = BertSoftmaxModel(cfg.bert_path, label_encoder)
        optimizer = Optimizer(model.all_parameters, steps=batch_num * config["epochs"])  # 优化器

        #　loss
        criterion = loss_factory.binarg_loss()
        best_train_f1, best_dev_f1 = 0, 0
        early_stop = -1
        EarlyStopEpochs = 10  # 当多个epoch，dev的指标都没有提升，则早停
        # train
        print("start train")
        for epoch in range(cfg.RESUME_EPOCH+1, config["epochs"] + 1):
            optimizer.zero_grad()
            model.train()  # 启用 BatchNormalization 和 Dropout
            overall_losses = 0
            losses = 0
            # batch_idx = 1
            y_pred = []
            y_true = []
            step = 0
            for batch_data in dataset_processer.data_iter(train_data, config["train_batch_size"], shuffle=True):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = dataset_processer.batch2tensor(batch_data)
                batch_outputs = model(batch_inputs)
                loss = criterion(batch_outputs, batch_labels)
                loss.backward()

                loss_value = loss.detach().cpu().item()
                losses += loss_value
                overall_losses += loss_value

                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

                # nn.utils.clip_grad_norm_(optimizer.all_params, max_norm=config["clip"])  # 梯度裁剪
                for cur_optim, scheduler in zip(optimizer.optims, optimizer.schedulers):
                    cur_optim.step()
                    scheduler.step()
                optimizer.zero_grad()
                step += 1
                # print(step, time.time())
            overall_losses /= batch_num
            overall_losses = scores.reformat(overall_losses, 4)
            score, train_f1 = scores.get_score(y_true, y_pred)
            print("epoch:{},train_score:{}, train_f1:{}, overall_loss:{} ".format(epoch, train_f1, score, overall_losses))
            _, dev_f1 = _eval(data=dev_data)

            if best_dev_f1 < dev_f1:
                best_dev_f1 = dev_f1
                early_stop = 0
                best_train_f1 = train_f1
                save_path = model_utils.save_checkpoint(
                    model, epoch, save_folder=os.path.join(cfg.proj_path, "data/bert_nn"))
                print("save_path:{}".format(save_path))
                # torch.save(model.state_dict(), save_model)
            else:
                early_stop += 1
                if early_stop == EarlyStopEpochs:  # 达到早停次数，则停止训练
                    break
            print("early_stop:{}, score:{}, dev_f1:{}, best_train_f1:{}, best_dev_f1:{}".format(
                early_stop, dev_f1, score, best_train_f1, best_dev_f1))


if __name__ == "__main__":
    run(mtd=MthStep.process_data.value)
