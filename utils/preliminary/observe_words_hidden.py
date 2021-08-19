from os import truncate
import torch
import json
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from transformers import BertModel

from tokenization import RawZhTokenizer

fpath = "/data/disk1/private/xcj/LegalBert/data/textual_data/ms/1.json"
data = json.load(open(fpath, "r"))

gpu = 0
layernum = 6
model_name = "bert-base-chinese"
# model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(gpu)
model = BertModel.from_pretrained("rawzh/raw_zh_ckpts/ckpt_18167.pt", output_hidden_states=True).to(gpu)
model.eval()
# tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = RawZhTokenizer.from_pretrained("rawzh/raw_zh_22675.vocab", "rawzh/raw_zh_22675.model")
token2sim = {}
for doc in tqdm(data[:10000]):
    inputs = tokenizer(doc, return_tensors="pt", max_length=512, truncation=True)
    for key in inputs:
        inputs[key] = inputs[key].to(gpu)
    outputs = model(**inputs)
    hiddens = outputs["hidden_states"] # tuple (13, FloatTensor: batch, seq_len, hidden_size)
    final_layer = hiddens[-1]
    final_length = torch.sqrt(torch.sum(final_layer * final_layer, dim=2))
    similarity = []
    for layer in range(layernum):
        low_layer = hiddens[layer] # batch, seq_len, hidden_size
        dot = torch.sum(final_layer * low_layer, dim=2)
        low_length = torch.sqrt(torch.sum(low_layer * low_layer, dim=2))
        sim = dot / (final_length * low_length)
        # from IPython import embed; embed()
        similarity.append(sim.squeeze(0))
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    for tid, token in enumerate(tokens):
        if token not in token2sim:
            token2sim[token] = {"times": 0, "sim": [0 for i in range(layernum)]}
        token2sim[token]["times"] += 1
        for i in range(layernum):
            token2sim[token]["sim"][i] += float(similarity[i][tid])

fout = open("token2sim.txt", "w")
for token in token2sim:
    for i in range(len(token2sim[token]["sim"])):
        token2sim[token]["sim"][i] = round(token2sim[token]["sim"][i] / token2sim[token]["times"], 4)
    print("%s\t%s" % (token, json.dumps(token2sim[token])), file = fout)
fout.close()

