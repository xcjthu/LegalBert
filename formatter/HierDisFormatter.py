import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, LongformerTokenizer
from formatter.SentTokenization import SentTokenization

class HierDisFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        # self.tokenizer = LongformerTokenizer.from_pretrained("/data/disk1/private/xcj/LegalBert/src/PLMConfig/roberta-converted-lfm/vocab.txt", pretrained_model_name_or_path='/data/disk1/private/xcj/LegalBert/src/PLMConfig/roberta-converted-lfm/')
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.mlm_prob = config.getfloat("train", "mlm_prob")
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_prob)
        self.shuffle_ratio = 0.2
        self.additional_tokenizer = SentTokenization()

    def process(self, data, config, mode, *args, **params):
        max_len = self.max_len
        inputx = []
        mask = np.zeros((len(data), max_len))
        sent_pos = []
        sent_ids = []
        for docid, docs in enumerate(data):
            doc = []
            spos = []
            sids = []
            for tmp in docs:
                tmp_ws, tmp_spos, tmp_sid = self.additional_tokenizer.add_sent_token(tmp, len(doc))
                doc += tmp_ws
                spos += tmp_spos
                sids += tmp_sid
            if len(doc) < max_len:
                mask[docid, :len(doc)] = 1
                inputx.append(torch.LongTensor( np.array(( doc + [self.tokenizer.pad_token_id] * ( self.max_len - len(doc) ) ).copy(), dtype=np.int16) ))
                sids += [0] * ( self.max_len - len(doc) )
            else:
                mask[docid] = 1
                inputx.append(torch.LongTensor( np.array(doc[:self.max_len].copy(), dtype=np.int16) ))
                spos = [pos for pos in spos if pos < max_len]
                sids = sids[:self.max_len]
            sent_pos.append(spos)
            sent_ids.append(sids)
        maxsent = max([len(sp) for sp in sent_pos])
        sent_pos = [sp + [-1] * (maxsent - len(sp)) for sp in sent_pos]

        ret = self.data_collator(inputx)
        ret['mask'] = torch.LongTensor(mask)

        ret["gat"] = np.zeros((len(data), self.max_len))
        ret["gat"][:,0] = 1
        ret["gat"] = torch.LongTensor(ret["gat"])
        ret["sent_pos"] = torch.LongTensor(sent_pos)
        ret["sent_ids"] = torch.LongTensor(sent_ids)
        return ret
