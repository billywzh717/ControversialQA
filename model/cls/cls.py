from datetime import datetime

import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils import data
from tqdm import tqdm
from transformers import AutoModel

from config import *


class MyDataset(data.Dataset):
    def __init__(self, data_path):
        self.data, self.labels = self.build_pair2(data_path)
        self.len = len(self.labels)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.labels[index], self.data[index]

    def build_pair(self, filepath, top=5):
        qa_pairs = []
        labels = []
        with open(filepath, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            lines = [line[:-1] for line in lines]
            for line in lines:
                ss = line.split('\t')
                question = ss[2]
                cons = ss[3:8]
                for con in cons[:top]:
                    qa_pairs.append(question + ' [SEP] ' + con)
                    labels.append(1)
                pros = ss[8:]
                for pro in pros[:top]:
                    qa_pairs.append(question + ' [SEP] ' + pro)
                    labels.append(0)
        return qa_pairs, labels

    def build_pair2(self, filepath):
        qa_pairs = []
        labels = []
        with open(filepath, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            lines = [line[:-1] for line in lines]
            for line in lines:
                ss = line.split('\t')
                question = ss[0]
                con = ss[1]
                qa_pairs.append(con + ' [SEP] ' + question)
                labels.append(1)
                pro = ss[2]
                qa_pairs.append(pro + ' [SEP] ' + question)
                labels.append(0)
        return qa_pairs, labels


class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.model = AutoModel.from_pretrained(config.model)
        self.tokenizer = config.tokenizer
        self.classifier = nn.Linear(in_features=config.hidden_size, out_features=2)

    def forward(self, sentence):
        tokens = self.tokenizer(sentence,
                                padding=True,
                                truncation=True,
                                max_length=512)
        text_ids = tokens['input_ids']
        attention_masks = tokens['attention_mask']

        text_ids = torch.tensor(text_ids, device=self.model.device)
        attention_masks = torch.tensor(attention_masks, device=self.model.device)
        feature = self.model(input_ids=text_ids,
                             attention_mask=attention_masks)

        feature = feature['pooler_output']  # [B, 768]
        output = self.classifier(feature)
        return output


def test(net, test_dataset):
    net.eval()
    with torch.no_grad():
        test_loader = data.DataLoader(test_dataset,
                                      batch_size=config.test_batch_size,
                                      shuffle=False,
                                      drop_last=False)
        ground_labels = []
        pre_labels = []

        for labels, sentences in tqdm(test_loader):
            labels = labels.to(config.device)
            sentences = list(sentences)

            output = net(sentences)
            cls = torch.argmax(output, dim=1)

            [ground_labels.append(label.item()) for label in labels]
            [pre_labels.append(label.item()) for label in cls]
    acc = accuracy_score(ground_labels, pre_labels)
    f1 = f1_score(ground_labels, pre_labels, average='macro')
    p = precision_score(ground_labels, pre_labels, average='macro')
    r = recall_score(ground_labels, pre_labels, average='macro')
    net.train()
    return acc, f1, p, r


def train(net):
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()
    cur_best_acc = 0

    for epoch in range(config.num_epoch):
        dataset = MyDataset(data_path=config.train_data_path)
        train_loader = data.DataLoader(dataset,
                                       batch_size=config.train_batch_size,
                                       shuffle=True,
                                       drop_last=True)
        index = 0
        batch_total_l = 0
        for labels, sentences in tqdm(train_loader):
            index += 1
            labels = labels.to(config.device)
            sentences = list(sentences)
            output = net(sentences)
            l = loss(output, labels)
            l = l / config.batch_accum
            l.backward()
            batch_total_l += l.item()

            if index % config.batch_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_total_l = 0
            torch.cuda.empty_cache()

        dev_dataset = MyDataset(data_path=config.dev_data_path)
        acc, f1, p, r = test(net, dev_dataset)
        print('dev acc', acc, 'f1:', f1, 'p:', p, 'r:', r, datetime.now())
        if acc > cur_best_acc:
            cur_best_acc = acc
            torch.save(net, config.model_save_path)
    return cur_best_acc


def evaluate(net):
    print('model name:', config.model)

    train_dataset = MyDataset(data_path=config.train_data_path)
    acc, f1, p, r = test(net, train_dataset)
    print('train acc:', acc, 'f1:', f1, 'p:', p, 'r:', r, datetime.now())

    dev_dataset = MyDataset(data_path=config.dev_data_path)
    acc, f1, p, r = test(net, dev_dataset)
    print('dev acc:', acc, 'f1:', f1, 'p:', p, 'r:', r, datetime.now())

    test_dataset = MyDataset(data_path=config.test_data_path)
    acc, f1, p, r = test(net, test_dataset)
    print('test acc:', acc, 'f1:', f1, 'p:', p, 'r:', r, datetime.now())


if __name__ == '__main__':

    # configs = [BertConfig(), BertLargeConfig(),
    #            RobertaConfig(), RobertaLargeConfig(),
    #            SpanBertConfig(), SpanBertLargeConfig()]

    configs = [RobertaConfig()]

    for config in configs:
        print('model name:', config.model, datetime.now())

        net = TextModel().to(device=config.device)
        cur_best_acc = train(net=net)
        print('dev best:', cur_best_acc, datetime.now())

        net = torch.load(config.model_save_path)

        train_dataset = MyDataset(data_path=config.train_data_path)
        acc, f1, p, r = test(net, train_dataset)
        print('train acc:', acc, 'f1:', f1, 'p:', p, 'r:', r, datetime.now())

        dev_dataset = MyDataset(data_path=config.dev_data_path)
        acc, f1, p, r = test(net, dev_dataset)
        print('dev acc:', acc, 'f1:', f1, 'p:', p, 'r:', r, datetime.now())

        test_dataset = MyDataset(data_path=config.test_data_path)
        acc, f1, p, r = test(net, test_dataset)
        print('test acc:', acc, 'f1:', f1, 'p:', p, 'r:', r, datetime.now())
