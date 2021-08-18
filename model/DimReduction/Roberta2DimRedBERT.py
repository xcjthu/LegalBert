from transformers import AutoModel,AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers.models import roberta
from DimRedBERT import DimRedBertModel, DimRedBertForMaskedLM
import torch
import copy
from torch import nn

config = AutoConfig.from_pretrained("hfl/chinese-roberta-wwm-ext", mirror="tuna")
roberta = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext", mirror="tuna")

max_pos = 2048
drconfig = AutoConfig.from_pretrained("hfl/chinese-roberta-wwm-ext", mirror="tuna")
# drconfig.num_attention_heads_layerwise = [12, 12, 12, 8, 8, 4, 4, 8, 8, 12, 12, 12]
# drconfig.hidden_size_layerwise = [768, 768, 768, 512, 512, 256, 256, 512, 512, 768, 768, 768]
drconfig.num_attention_heads_layerwise = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
# drconfig.hidden_size_layerwise = [768, 768, 768, 384, 384, 192, 192, 384, 384, 768, 768, 768]
drconfig.hidden_size_layerwise = [768, 768, 384, 384, 384, 192, 192, 384, 384, 384, 768, 768]
# DRBert = DimRedBertForMaskedLM(drconfig)
current_max_pos, embed_size = roberta.bert.embeddings.position_embeddings.weight.shape
drconfig.max_position_embeddings = max_pos

new_pos_embed = roberta.bert.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
k = 2
step = current_max_pos - 2
while k < max_pos - 1:
    if k + step >= max_pos:
        new_pos_embed[k:] = roberta.bert.embeddings.position_embeddings.weight[2:(max_pos + 2 - k)]
    else:
        new_pos_embed[k:(k + step)] = roberta.bert.embeddings.position_embeddings.weight[2:]
    k += step
roberta.bert.embeddings.position_embeddings.weight.data = new_pos_embed
roberta.bert.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)
for i in range(len(roberta.bert.encoder.layer)):
    print("converting layer", i, "...")
    if i != 11 and drconfig.hidden_size_layerwise[i] != drconfig.hidden_size_layerwise[i + 1]:
        roberta.bert.encoder.layer[i].output.changesize = nn.Linear(drconfig.hidden_size_layerwise[i], drconfig.hidden_size_layerwise[i + 1])
    if drconfig.hidden_size_layerwise[i] == 768:
        continue
    else:
        hsize = drconfig.hidden_size_layerwise[i]
        rate = 768 // hsize
        def compress(linear):
            ret = nn.Linear(hsize, hsize)
            ret.weight.data.zero_()
            ret.bias.data.zero_()
            for row in range(rate):
                for col in range(rate):
                    ret.weight.data += linear.weight.data[row * hsize : (row + 1) * hsize, col * hsize : (col + 1) * hsize]
                ret.bias.data += linear.bias.data[row * hsize : (row + 1) * hsize]
            ret.weight.data /= rate
            ret.bias.data /= rate
            return ret
        # Attention
        for attr in ["query", "key", "value"]:
            # self Attention
            ori = getattr(roberta.bert.encoder.layer[i].attention.self, attr) # 768, 768
            setattr(roberta.bert.encoder.layer[i].attention.self, attr, compress(ori))
        # Attention output
        roberta.bert.encoder.layer[i].attention.output.dense = compress(roberta.bert.encoder.layer[i].attention.output.dense)
        # LayerNorm
        def compress_ln(layernorm):
            ret = nn.LayerNorm(hsize, config.layer_norm_eps)
            ret.weight.data.zero_()
            ret.bias.data.zero_()
            for row in range(rate):
                ret.weight.data += layernorm.weight.data[row * hsize : (row + 1) * hsize]
                ret.bias.data += layernorm.bias.data[row * hsize : (row + 1) * hsize]
            return ret
        roberta.bert.encoder.layer[i].attention.output.LayerNorm = compress_ln(roberta.bert.encoder.layer[i].attention.output.LayerNorm)

        def compress_intermediate(intermediate):
            ret = nn.Linear(hsize, hsize * 4)
            ret.weight.data.zero_()
            ret.bias.data.zero_()
            for row in range(rate):
                for col in range(rate):
                    ret.weight.data += intermediate.weight.data[4 * row * hsize : 4 * (row + 1) * hsize, col * hsize : (col + 1) * hsize]
                ret.bias.data += intermediate.bias.data[4 * row * hsize : 4 * (row + 1) * hsize]
            ret.weight.data /= rate
            ret.bias.data /= rate
            return ret
        roberta.bert.encoder.layer[i].intermediate.dense = compress_intermediate(roberta.bert.encoder.layer[i].intermediate.dense)

        roberta.bert.encoder.layer[i].output.LayerNorm = compress_ln(roberta.bert.encoder.layer[i].output.LayerNorm)
        def compress_output_dense(intermediate):
            ret = nn.Linear(hsize * 4, hsize)
            ret.weight.data.zero_()
            ret.bias.data.zero_()
            for row in range(rate):
                for col in range(rate):
                    ret.weight.data += intermediate.weight.data[row * hsize : (row + 1) * hsize, 4 * col * hsize : 4 * (col + 1) * hsize]
                ret.bias.data += intermediate.bias.data[row * hsize : (row + 1) * hsize]
            ret.weight.data /= rate
            ret.bias.data /= rate
            return ret
        roberta.bert.encoder.layer[i].output.dense = compress_output_dense(roberta.bert.encoder.layer[i].output.dense)
roberta.config = drconfig
roberta.save_pretrained("/home/xcj/LegalLongPLM/LegalLongPLM/PLMConfig/DimRedBERT")
drbert = DimRedBertModel.from_pretrained("/home/xcj/LegalLongPLM/LegalLongPLM/PLMConfig/DimRedBERT")
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", mirror="tuna")
inp = tokenizer("我爱北京天安门", return_tensors="pt")
output = drbert(**inp)
from IPython import embed; embed()
