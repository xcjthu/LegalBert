import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, LongformerTokenizer

class UnContrastiveFormatterVer2(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        # self.tokenizer = LongformerTokenizer.from_pretrained("/data/disk1/private/xcj/LegalBert/src/PLMConfig/roberta-converted-lfm/vocab.txt", pretrained_model_name_or_path='/data/disk1/private/xcj/LegalBert/src/PLMConfig/roberta-converted-lfm/')
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.mlm_prob = config.getfloat("train", "mlm_prob")
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_prob)
        self.shuffle_ratio = 0.3

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

    def shuffle_part_doc(self, doc):
        # doc: doc_len (list)
        sents = []
        last_pos = 1
        for tpos, token in enumerate(doc[1:-1]):
            if token == 511:
                sents.append(doc[last_pos:tpos + 2])
                last_pos = tpos + 2
        sents.append(doc[last_pos:-1])
        # if len(sents) < 3:
        #     return doc.tolist()
        selected = random.sample(list(range(len(sents))), int(len(sents) * self.shuffle_ratio))
        shuffle = {i: i for i in range(len(sents))}
        for i in range(-1, len(selected) - 1):
            shuffle[selected[i]] = selected[i + 1]
        ret = [int(doc[0])]
        for sid, sent in enumerate(sents):
            ret += sents[shuffle[sid]].tolist()
        # print(self.tokenizer.decode(ret))
        # print(self.tokenizer.decode(doc))
        # print("==" * 20)
        ret.append(int(doc[-1]))
        if len(ret) != len(doc):
            print("shuffle:", shuffle)
            print("selected:", selected)
            print("sents:", sents)
            print("docs:", doc)
        assert len(ret) == len(doc)
        return ret

    def process(self, data, config, mode, *args, **params):
        max_len = self.max_len
        inputx = []
        data.append([np.array(self.shuffle_part_doc(doc)) for doc in data[0]])
        mask = np.zeros((len(data), max_len))
        for docid, docs in enumerate(data):
            doc = []
            for tmp in docs:
                doc += tmp.tolist()
            if len(doc) < max_len:
                mask[docid, :len(doc)] = 1
                inputx.append(torch.LongTensor( np.array(( doc + [self.tokenizer.pad_token_id] * ( self.max_len - len(doc) ) ).copy(), dtype=np.int16) ))
            else:
                mask[docid] = 1
                inputx.append(torch.LongTensor( np.array(doc[:self.max_len].copy(), dtype=np.int16) ))
        ret = self.data_collator(inputx)
        ret['mask'] = torch.LongTensor(mask)

        ret["gat"] = np.zeros((len(data), self.max_len))
        ret["gat"][:,0] = 1
        ret["gat"] = torch.LongTensor(ret["gat"])
        return ret
