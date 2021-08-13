from transformers import AutoModelForMaskedLM,AutoModelForPreTraining,LongformerConfig,LongformerForMaskedLM
import torch
from torch import nn
from .DimReduction.DimRedBERT import DimRedBertForMaskedLM

class VanillaLFM(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(VanillaLFM, self).__init__()
        # config = LongformerConfig.from_pretrained('/mnt/datadisk0/xcj/LegalBert/LegalBert/PLMConfig/roberta-converted-lfm')
        # self.LFM = LongformerForMaskedLM(config)
        self.LFM = LongformerForMaskedLM.from_pretrained('/mnt/datadisk0/xcj/LegalBert/LegalBert/PLMConfig/roberta-converted-lfm')

    def save_pretrained(self, path):
        self.LFM.save_pretrained(path)

    def forward(self, data, config, gpu_list, acc_result, mode):
        ret = self.LFM(data['input_ids'], attention_mask=data['mask'], labels=data['labels'])
        loss, logits = ret[0], ret[1]
        return {"loss": loss, "acc_result":{}}

class VanillaBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(VanillaBert, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def save_pretrained(self, path):
        self.bert.save_pretrained(path)
    
    def forward(self, data, config, gpu_list, acc_result, mode):
        ret = self.bert(data['input_ids'], attention_mask=data['mask'], labels=data['labels'])
        loss, logits = ret[0], ret[1]
        return {"loss": loss, "acc_result": {}}

class VanillaDimRedBERT(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(VanillaDimRedBERT, self).__init__()
        self.bert = DimRedBertForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def save_pretrained(self, path):
        self.bert.save_pretrained(path)

    def forward(self, data, config, gpu_list, acc_result, mode):
        ret = self.bert(data['input_ids'], attention_mask=data['mask'], labels=data['labels'])
        loss, logits = ret[0], ret[1]
        return {"loss": loss, "acc_result": {}}
