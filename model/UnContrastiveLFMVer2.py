from transformers import AutoModelForMaskedLM,AutoModelForPreTraining,LongformerConfig,LongformerForMaskedLM
import torch
from torch import nn
import torch.distributed as dist
from model.LongformerSentID.LongformerSentID import LongformerSentIDForMaskedLM

class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class UnContrastiveLFMVer2(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(UnContrastiveLFMVer2, self).__init__()
        # config = LongformerConfig.from_pretrained('/mnt/datadisk0/xcj/LegalBert/LegalBert/PLMConfig/roberta-converted-lfm')
        # self.LFM = LongformerForMaskedLM(config)
        # self.LFM = LongformerForMaskedLM.from_pretrained('thunlp/Lawformer', output_hidden_states=True)
        self.LFM = LongformerSentIDForMaskedLM.from_pretrained('thunlp/Lawformer', output_hidden_states=True)
        self.pooler = Pooler(self.LFM.config)
        self.sim = nn.CosineSimilarity(dim=-1)
        self.loss2 = nn.CrossEntropyLoss()

    def save_pretrained(self, path):
        self.LFM.save_pretrained(path)

    def forward(self, data, config, gpu_list, acc_result, mode):
        out = self.LFM(data['input_ids'], attention_mask=data['mask'], global_attention_mask=data["gat"], labels=data['labels'])
        loss, hiddens = out["loss"], out["hidden_states"][-1]
        ret = self.pooler(hiddens)

        neg = ret[:-1] # batch, hidden_size
        pos = ret[-1].unsqueeze(0) #  1, hidden_size
        if dist.is_initialized() and self.training:
            neg_list = [torch.zeros_like(neg) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=neg_list, tensor=neg.contiguous())
            neg_list[dist.get_rank()] = neg
            neg = torch.cat(neg_list, dim=0) # batch * world_size, hidden_size

            pos_list = [torch.zeros_like(pos) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=pos_list, tensor=pos.contiguous())
            pos_list[dist.get_rank()] = pos
            pos = torch.cat(pos_list, dim=0) # world_size, hidden_size
        sim = self.sim(pos.unsqueeze(1), neg.unsqueeze(0))
        label = torch.arange(sim.shape[0]).to(sim.device)
        loss2 = self.loss2(sim, label)
        return {"loss": loss + loss2, "acc_result":{}}
