from transformers import AutoModelForMaskedLM,AutoModelForPreTraining,LongformerConfig,LongformerForMaskedLM
import torch
from torch import nn
from transformers.models.longformer.modeling_longformer import LongformerPooler
from .metric import softmax_acc

class ContrastiveLawLFM(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ContrastiveLawLFM, self).__init__()
        self.LFM = LongformerForMaskedLM.from_pretrained('/mnt/datadisk0/xcj/LegalBert/LegalBert/PLMConfig/roberta-converted-lfm')
        self.pooler = LongformerPooler(self.LFM.config)

        self.hidde_size = self.LFM.config.hidden_size
        self.cross_attention = nn.MultiheadAttention(self.hidde_size, 1)
        self.outlinear = nn.Linear(self.hidde_size, 1)
        self.lossfn = nn.CrossEntropyLoss()

    def save_pretrained(self, path):
        self.LFM.save_pretrained(path)

    def compute_score(self, lhid, fhid):
        # lhid: cand_num, law_len, hidde_size
        # fhid: batch, fact_len, hidden_size
        batch, cand_num = fhid.shape[0], lhid.shape[0]
        lhids = lhid.unsqueeze(0).repeat(batch, 1, 1, 1).view(batch * cand_num, lhid.shape[1], self.hidde_size) # batch * cand_num, law_len, hidden_size
        fhids = fhid.unsqueeze(1).repeat(1, cand_num, 1, 1).view(batch * cand_num, fhid.shape[1], self.hidde_size) # batch * cand_num, f_len, hidden_size
        att_out, _ = self.cross_attention(torch.transpose(lhids, 0, 1), torch.transpose(fhids, 0, 1), torch.transpose(fhids, 0, 1)) # batch * cand_num, law_len, hidden_size
        att_out = torch.transpose(att_out, 0, 1)
        feature, _ = torch.max(att_out, dim = 1) # batch * cand_num, hidden_size
        #print(lhids.shape, fhids.shape, att_out.shape, feature.shape)
        score = self.outlinear(feature).view(batch, cand_num)
        return score

    def forward(self, data, config, gpu_list, acc_result, mode):
        ret = self.LFM(data['input_ids'], attention_mask=data['mask'], labels=data['labels'], output_hidden_states=True)
        loss, logits, fact_hiddens = ret[0], ret[1], ret[2][0]
        lret = self.LFM(data['laws'], attention_mask=data['law_mask'], output_hidden_states=True)
        law_hidden = lret[1][0]

        match_score = self.compute_score(law_hidden, fact_hiddens)

        loss += self.lossfn(match_score, data['law_labels'])
        acc_result = softmax_acc(match_score, data['law_labels'], acc_result)
        return {"loss": loss, "acc_result": acc_result}

