import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, LongformerTokenizer

class UnContrastiveFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        # self.tokenizer = LongformerTokenizer.from_pretrained("/data/disk1/private/xcj/LegalBert/src/PLMConfig/roberta-converted-lfm/vocab.txt", pretrained_model_name_or_path='/data/disk1/private/xcj/LegalBert/src/PLMConfig/roberta-converted-lfm/')
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.mlm_prob = config.getfloat("train", "mlm_prob")
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_prob)
 
    def shuffle_doc(self, doc):
        # doc: doc_len
        sents = []
        last_pos = 1
        for tpos, token in enumerate(doc[1:-1]):
            if token == 511:
                sents.append(doc[last_pos:tpos + 2])
                last_pos = tpos + 2
        sents.append(doc[last_pos:-1])
        random.shuffle(sents)
        ret = [int(doc[0])]
        for sent in sents:
            ret += sent.tolist()
        ret.append(int(doc[-1]))
        # print("==" * 20)
        # print(self.tokenizer.decode(doc))
        # print(self.tokenizer.decode(ret))
        return ret

    def process(self, data, config, mode, *args, **params):
        max_len = self.max_len
        inputx = []
        data.append([self.shuffle_doc(doc) for doc in data[0]])
        mask = np.zeros((len(data), max_len))
        for docid, doc in enumerate(data):
            if len(doc) < max_len:
                mask[docid, :len(doc)] = 1
                inputx.append(torch.LongTensor( np.array(( doc.tolist() + [self.tokenizer.pad_token_id] * ( max_len - len(doc) ) ).copy(), dtype=np.int16) ))
            else:
                mask[docid] = 1
                inputx.append(torch.LongTensor( np.array(doc[:max_len].copy(), dtype=np.int16) ))
        ret = self.data_collator(inputx)
        ret['mask'] = torch.LongTensor(mask)

        ret["gat"] = np.zeros((len(data), self.max_len))
        ret["gat"][:,0] = 1
        ret["gat"] = torch.LongTensor(ret["gat"])
        return ret