import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, LongformerTokenizer
from dataset.LawDataset import LawDataset

class LawClassifierFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.mode = mode
        self.max_len = config.getint("train", "max_len")

        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.mlm_prob = config.getfloat("train", "mlm_prob")
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_prob)
        self.lawdata = LawDataset(config)


    def process(self, data, config, mode, *args, **params):
        docs = [torch.LongTensor(doc['doc'].tolist()) for doc in data]
        ret = self.data_collator(docs)

        max_len = min(self.max_len, max([len(doc['doc']) for doc in data]))
        mask = np.zeros((len(data), max_len))
        for docid, doc in enumerate(data):
            mask[docid, : min(len(doc), max_len)] = 1

        labels = [i if data[i]['label'] else -100 for i in range(len(data))]

        cands = [doc['cand'] for doc in data]
        all_laws = set([l for doc in data for l in doc['all_law']])
        cands += self.lawdata.sample_negative_more(all_laws, self.law_neg_num)

        law_max_len = min(self.law_max_len, max([len(law) for law in cands]))
        law_mask = np.zeros((len(cands), law_max_len))
        for lid, law in enumerate(cands):
            law_mask[lid, :min(len(law), law_max_len)] = 1

        ret['input_ids'] = ret['input_ids'][:,:max_len]
        ret['labels'] = ret['labels'][:,:max_len]
        ret['mask'] = torch.LongTensor(mask)

        ret['law_labels'] = torch.LongTensor(labels)
        ret['law_mask'] = torch.LongTensor(law_mask)
        ret['laws'] = torch.LongTensor([law + [self.tokenizer.sep_token_id] * (law_max_len - len(law)) if len(law) < law_max_len else law[:law_max_len] for law in cands])

        return ret
