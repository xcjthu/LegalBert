from transformers import AutoModelForMaskedLM,AutoModelForPreTraining,LongformerConfig,LongformerForMaskedLM
import torch
from torch import nn

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
        self.LFM = LongformerForMaskedLM.from_pretrained('thunlp/Lawformer', output_hidden_states=True)
        self.pooler = Pooler(self.LFM.config)
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
        score = orep.mm(torch.transpose(rrep, 0, 1)) # batch, batch
        label = torch.arange(batch).to(score.device)
        loss2 = self.loss2(score, label)
        return {"loss": loss + loss2, "acc_result":{}}
