import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, LongformerTokenizer
from formatter.SentTokenization import SentTokenization

class RecurrentTransFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        # self.max_len = config.getint("train", "max_len")
        # self.pred_len = config.getint("train", "pred_len")
        self.mode = mode
        # self.tokenizer = LongformerTokenizer.from_pretrained("/data/disk1/private/xcj/LegalBert/src/PLMConfig/roberta-converted-lfm/vocab.txt", pretrained_model_name_or_path='/data/disk1/private/xcj/LegalBert/src/PLMConfig/roberta-converted-lfm/')
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.mlm_prob = config.getfloat("train", "mlm_prob")
        self.pred_mlm_prob = config.getfloat("train", "pred_mlm_prob")
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_prob)
        self.pred_data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.pred_mlm_prob)
        self.sent_tokenizer = SentTokenization(config)
        self.block_len = config.getint("train", "block_len")
        self.max_block_size = config.getint("train", "block_num")

    def process(self, data, config, mode, *args, **params):
        inpblocks = []
        pred_block = []
        mask = []
        pred_mask = []
        for docid, docs in enumerate(data):
            inpb, predb, tmask, tpmask = self.sent_tokenizer.split_for_recurrent(docs, block_len=self.block_len, pad_id=self.tokenizer.pad_token_id, max_block_size=self.max_block_size)
            assert len(inpb) == self.max_block_size
            inpblocks.append(inpb), pred_block.append(predb)
            mask.append(tmask), pred_mask.append(tpmask)
        inpblocks = list(torch.LongTensor(inpblocks).view(-1, self.block_len))
        inpb_out = self.data_collator(inpblocks)
        predb_out = self.pred_data_collator(pred_block)

        return {
            "inpb_inp": inpb_out["input_ids"].view(len(data), self.max_block_size, self.block_len),
            "inpb_mlm": inpb_out["labels"].view(len(data), self.max_block_size, self.block_len),
            "inpb_mask": torch.LongTensor(mask),

            "predb_inp": predb_out["input_ids"],
            "predb_mlm": predb_out["labels"],
            "predb_mask": torch.LongTensor(pred_mask),
        }
