import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, LongformerTokenizer

class VanillaLFMFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        # self.tokenizer = LongformerTokenizer.from_pretrained("/data/disk1/private/xcj/LegalBert/src/PLMConfig/roberta-converted-lfm/vocab.txt", pretrained_model_name_or_path='/data/disk1/private/xcj/LegalBert/src/PLMConfig/roberta-converted-lfm/')
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.mlm_prob = config.getfloat("train", "mlm_prob")
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_prob)

    def process(self, data, config, mode, *args, **params):
        max_len = min([self.max_len] + [len(inp) for inp in data])
        inputx = []
        mask = np.zeros((len(data), max_len))
        for docid, doc in enumerate(data):
            if len(doc) < max_len:
                mask[docid, :len(doc)] = 1
                inputx.append(torch.LongTensor( np.array(( doc + [self.tokenizer.pad_token_id] * (max_len - len(doc) ) ).copy(), dtype=np.int16) ))
            else:
                mask[docid] = 1
                inputx.append(torch.LongTensor( np.array(doc[:max_len].copy(), dtype=np.int16) ))
        ret = self.data_collator(inputx)
        ret['mask'] = torch.LongTensor(mask)
        return ret
