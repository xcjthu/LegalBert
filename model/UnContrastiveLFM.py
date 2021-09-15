from transformers import AutoModelForMaskedLM
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

class UnContrastiveLFM(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(UnContrastiveLFM, self).__init__()
        # config = LongformerConfig.from_pretrained('/mnt/datadisk0/xcj/LegalBert/LegalBert/PLMConfig/roberta-converted-lfm')
        # self.LFM = LongformerForMaskedLM(config)
        # self.LFM = AutoModelForMaskedLM.from_pretrained('thunlp/Lawformer', output_hidden_states=True)
        self.LFM = LongformerSentIDForMaskedLM.from_pretrained('thunlp/Lawformer', output_hidden_states=True)
        self.pooler = Pooler(self.LFM.config)
        self.sim = nn.CosineSimilarity(dim=-1)
        self.loss2 = nn.CrossEntropyLoss()

    def save_pretrained(self, path):
        self.LFM.save_pretrained(path)

    def forward(self, data, config, gpu_list, acc_result, mode):
        batch = data["input_ids"].shape[0]
        input_ids = torch.cat([data["input_ids"], data["r-input_ids"]], dim=0)
        attention_mask = torch.cat([data["mask"], data["r-mask"]], dim=0)
        labels = torch.cat([data["labels"], data["r-labels"]], dim=0)
        # print(data["gat"].device, attention_mask.device)
        ret = self.LFM(input_ids, attention_mask=attention_mask, global_attention_mask=data["gat"], labels=labels)
        loss, hiddens = ret["loss"], ret["hidden_states"][-1]
        clsh = self.pooler(hiddens)
        orep, rrep = clsh[:batch], clsh[batch:] # batch, hidden_size
        # score = orep.mm(torch.transpose(rrep, 0, 1)) # batch, batch
        if dist.is_initialized() and self.training:
            orep_list = [torch.zeros_like(orep) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=orep_list, tensor=orep.contiguous())
            orep_list[dist.get_rank()] = orep
            orep = torch.cat(orep_list, dim=0) # batch * world_size, hidden_size

            rrep_list = [torch.zeros_like(rrep) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=rrep_list, tensor=rrep.contiguous())
            rrep_list[dist.get_rank()] = rrep
            rrep = torch.cat(rrep_list, dim=0) # world_size, hidden_size
        score = self.sim(orep.unsqueeze(1), rrep.unsqueeze(0))
        label = torch.arange(score.shape[0]).to(score.device)
        loss2 = self.loss2(score, label)
        return {"loss": loss + loss2, "acc_result":{}}
