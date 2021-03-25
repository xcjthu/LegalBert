import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, LongformerTokenizer

class DocLawFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.max_len = config.getint("train", "max_len")
        self.mode = mode

        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.mlm_prob = config.getfloat("train", "mlm_prob")
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_prob)

    def process(self, data, config, mode, *args, **params):
        docs = [torch.LongTensor(doc['doc'].tolist()) for doc in data]
        mlmret = self.data_collator(docs)

        max_len = min(self.max_len, max([len(doc['doc']) + len(doc['gtlaw']) + len(doc['neglaw']) + 2 for doc in data]))
        inputx = np.zeros((len(data), max_len))
        mlmlabels = np.zeros((len(data), max_len)) - 100
        mask = np.zeros((len(data), max_len))
        global_att = np.zeros((len(data), max_len))
        global_att[:,0] = 1
        lawlabel = []
        for docid, doc in enumerate(data):
            if random.random() < 0.5:
                lcont = doc['gtlaw'] + [self.tokenizer.sep_token_id] + doc['neglaw'] + [self.tokenizer.sep_token_id]
                label = 0
            else:
                lcont = doc['neglaw'] + [self.tokenizer.sep_token_id] + doc['gtlaw'] + [self.tokenizer.sep_token_id]
                label = 1
            if len(lcont) + len(doc['doc']) > self.max_len:
                redundant = len(lcont) + len(doc['doc']) - self.max_len
                if redundant > 0.5 * len(doc['doc']): # 如果大部分事实内容都没办法保留，那就不做法条分类了
                    lcont = []
                    label = -100
                    if len(doc['doc']) >= max_len:
                        inputx[docid] = mlmret['input_ids'][docid, :max_len]
                        mlmlabels[docid] = mlmret['labels'][docid, :max_len]
                else: # 切掉部分事实内容来做法条分类
                    # doc['doc'] = doc['doc'][:-redundant]
                    doclen = len(doc['doc']) - redundant
                    #doc['doc'][-1] = self.tokenizer.sep_token_id
                    
                    inputx[docid,:doclen - 1] = mlmret['input_ids'][docid, : doclen - 1]
                    inputx[docid,doclen-1] = self.tokenizer.sep_token_id
                    mlmlabels[docid,:doclen - 1] = mlmret['labels'][docid,:doclen - 1]
                    inputx[docid, doclen:] = lcont
                    global_att[docid, doclen:] = 1
                mask[docid] = 1
            else:
                inputx[docid,:len(doc['doc'])] = mlmret['input_ids'][docid,:len(doc['doc'])]
                mlmlabels[docid,:len(doc['doc'])] = mlmret['labels'][docid,:len(doc['doc'])]
                inputx[docid,len(doc['doc']):len(doc['doc']) + len(lcont)] = lcont
                global_att[docid,len(doc['doc']):len(doc['doc']) + len(lcont)] = 1
            lawlabel.append(label)

        ret = {
            'input_ids': torch.LongTensor(inputx),
            'mlmlabels': torch.LongTensor(mlmlabels),
            'mask': torch.LongTensor(mask),
            'label': torch.LongTensor(lawlabel),
            'global_att': torch.LongTensor(global_att),
        }
        return ret
