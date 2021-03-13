from transformers import AutoModelForMaskedLM,AutoModelForPreTraining
import torch
from torch import nn

class VanillaLFM(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(VanillaLFM, self).__init__()

        self.LFM = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

    def forward(self, data, config, gpu_list, acc_result, mode):
        ret = self.LFM(data['input_ids'], attention_mask=data['mask'], labels=data['labels'])
        loss, logits = ret[0], ret[1]
        return {"loss": loss, "acc_result":{}}