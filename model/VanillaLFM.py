from transformers import AutoModelForMaskedLM,AutoModelForPreTraining,LongformerConfig,LongformerForMaskedLM
import torch
from torch import nn

class VanillaLFM(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(VanillaLFM, self).__init__()
        #config = LongformerConfig.from_pretrained('/data/disk1/private/xcj/LegalBert/src/PLMConfig/LFM.config')
        #self.LFM = LongformerForMaskedLM(config)
        self.LFM = LongformerForMaskedLM.from_pretrained('/data/xcj/LegalBert/LegalBert/PLMConfig/roberta-converted-lfm/')
        # self.LFM = LongformerForMaskedLM.from_pretrained('schen/longformer-chinese-base-4096', config=config)#AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
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
