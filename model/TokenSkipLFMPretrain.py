from transformers import AutoModelForMaskedLM,AutoModelForPreTraining,LongformerConfig,LongformerForMaskedLM
import torch
from torch import nn
from .DimReduction.DimRedBERT import DimRedBertForMaskedLM
from .TokenSkip.TokenSkipLFM import TokenSkipLFMForMaskedLM

class TokenSkipLFMPretrain(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(TokenSkipLFMPretrain, self).__init__()
        # config = LongformerConfig.from_pretrained('/mnt/datadisk0/xcj/LegalBert/LegalBert/PLMConfig/roberta-converted-lfm')
        # self.LFM = LongformerForMaskedLM(config)
        self.LFM = TokenSkipLFMForMaskedLM.from_pretrained('thunlp/Lawformer')

    def save_pretrained(self, path):
        self.LFM.save_pretrained(path)

    def forward(self, data, config, gpu_list, acc_result, mode):
        ret = self.LFM(data['input_ids'], attention_mask=data['mask'], labels=data['labels'])
        loss, logits = ret[0], ret[1]
        return {"loss": loss, "acc_result":{}}
