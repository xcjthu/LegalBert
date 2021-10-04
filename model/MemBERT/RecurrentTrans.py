from model.MemBERT.MemBERT import MemBertForMaskedLM
from torch import nn
import torch
from transformers import AutoConfig

class RecurrentTransMLM(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super().__init__()
        self.mlm_config = AutoConfig.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.mlm_config.mem_size = config.getint("train", "mem_size")
        self.encoder = MemBertForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext", config=self.mlm_config)

    def save_pretrained(self, path):
        self.encoder.save_pretrained(path)

    def forward(self, data, config, gpu_list, acc_result, mode):
        # data["inpb_inp"], data["inpb_mlm"]: batch, block_num, max_len
        # data["predb_inp"], data["predb_mlm"]: batch, max_len
        mem = None
        inpb_inp, inpb_mlm, inpb_mask = torch.transpose(data["inpb_inp"], 0, 1).contiguous(), torch.transpose(data["inpb_mlm"], 0, 1).contiguous(), torch.transpose(data["inpb_mask"], 0, 1).contiguous()
        block_num = inpb_inp.shape[0]
        tloss = 0
        for i in range(block_num):
            out = self.encoder(inpb_inp[i], attention_mask=inpb_mask[i], mem=mem, labels=inpb_mlm[i], return_dict=False)
            loss, mlm_logits, mem = out[0], out[1], out[2]
            tloss = tloss + loss
        out_pred = self.encoder(data["predb_inp"], attention_mask=data["predb_mask"], mem=mem, labels=data["predb_mlm"], return_dict=False)
        loss2, mem = out_pred[0], out_pred[2]
        tloss = tloss / block_num + loss2
        return {"loss": tloss, "acc_result": {}}
