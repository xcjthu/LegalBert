from transformers import AutoModelForMaskedLM,AutoModelForPreTraining,LongformerConfig,LongformerForMaskedLM
import torch
from torch import nn
from transformers.models.longformer.modeling_longformer import LongformerPooler

class DocLawLFM(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(DocLawLFM, self).__init__()
        self.LFM = LongformerForMaskedLM.from_pretrained('/mnt/datadisk0/xcj/LegalBert/LegalBert/PLMConfig/roberta-converted-lfm')
        self.pooler = LongformerPooler(self.LFM.config)
        self.outlinear = nn.Linear(self.LFM.config.hidden_size, 2)
        self.lossfn = nn.CrossEntropyLoss()
        # self.LFM = LongformerForMaskedLM.from_pretrained('schen/longformer-chinese-base-4096', config=config)#AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
    def save_pretrained(self, path):
        self.LFM.save_pretrained(path)

    def forward(self, data, config, gpu_list, acc_result, mode):
        ret = self.LFM(data['input_ids'], attention_mask=data['mask'], global_attention_mask=data['global_att'], labels=data['mlmlabels'], output_hidden_states=True)
        loss, logits, hiddens = ret[0], ret[1], ret[2]
        bcls = self.pooler(hiddens[0])
        score = self.outlinear(bcls)
        loss += self.lossfn(score, data['label'])
        return {"loss": loss, "acc_result":{}}

