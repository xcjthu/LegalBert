from os import truncate
import torch
import json
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

fpath = "/data/disk1/private/xcj/LegalBert/data/textual_data/xs/1.json"
data = json.load(open(fpath, "r"))

gpu = 0
model_name = "bert-base-chinese"
model = AutoModel.from_pretrained(model_name, output_attentions=True).to(gpu)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
token2score = {}
for doc in tqdm(data[:1000]):
    inputs = tokenizer(doc, return_tensors="pt", max_length=512, truncation=True)
    for key in inputs:
        inputs[key] = inputs[key].to(gpu)
    outputs = model(**inputs)

    attention = outputs["attentions"] # tuple (12, FloatTensor: batch, head_num, seq_len, seq_len)
    attention = [torch.mean(attma.squeeze(0), dim=0) for attma in attention]
    # attention = [torch.mean(attma - torch.diag_embed(torch.diag(attma)), dim=0) for attma in attention]
    attention = [torch.diag(attma) for attma in attention]
    # from IPython import embed; embed()
    # attention = [torch.mean(torch.mean(attma, dim=2), dim=1).squeeze(0) for attma in attention] # list 12 * FloatTensor: seq_len
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    for tid, token in enumerate(tokens):
        for layer in range(12):
            if token not in token2score:
                token2score[token] = {"times": 0, "score": [0 for i in range(12)]}
            token2score[token]["score"][layer] += float(attention[layer][tid])
        token2score[token]["times"] += 1


fout = open("token2self_score.txt", "w")
for token in token2score:
    for i in range(len(token2score[token]["score"])):
        token2score[token]["score"][i] = round(1000 * token2score[token]["score"][i] / token2score[token]["times"], 4)
    print("%s\t%s" % (token, json.dumps(token2score[token])), file = fout)
fout.close()

