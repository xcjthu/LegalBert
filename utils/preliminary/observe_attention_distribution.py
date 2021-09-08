# 观察attention随着距离的分布，不出意外肯定是local信息比较重要
# 观察hidden相似性和距离的关系，不出意外应该也是hidden相似性是距离越近越高
# 那么假设，到了第k层之后，强制要求模型attend到远的地方，可否增加全局信息的获取
# 这个实验是不是可以用DocRED来做

import torch
import json
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

fpath = "/home/xcj/LegalLongPLM/data/DocRED/dev.json"
data = json.load(open(fpath, "r"))
data = [" ".join([" ".join(sent) for sent in doc["sents"]]) for doc in data]

# gpu = 5
# layernum = 12
# # model_name = "bert-base-chinese"
# model_name = "bert-base-cased"
# model = AutoModel.from_pretrained(model_name, output_attentions=True).to(gpu)
# model.eval()
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# token2sim = {}
# pos2att = [{} for i in range(layernum)]
# for doc in tqdm(data[:100]):
#     # batch = 1
#     inputs = tokenizer(doc, return_tensors="pt", max_length=512, truncation=True)
#     for key in inputs:
#         inputs[key] = inputs[key].to(gpu)
#     outputs = model(**inputs)
    
#     attention = outputs["attentions"] # tuple (12, FloatTensor: batch, head_num, seq_len, seq_len)
#     # from IPython import embed; embed()
#     attention = [att.squeeze(0).mean(dim=0) for att in attention] # list (12, FloatTensor: seq_len, seq_len)
#     seq_len = attention[0].shape[0]
#     for layer in range(layernum):
#         for i in range(1, seq_len - 1): # 不想看special Token
#             for j in range(1, seq_len - 1):
#                 dis = i - j
#                 if not dis in pos2att[layer]:
#                     pos2att[layer][dis] = {"score": 0.0, "times": 0}
#                 pos2att[layer][dis]["times"] += 1
#                 pos2att[layer][dis]["score"] += float(attention[layer][i,j])

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# fout = open("attention_distribution.json", "w")
# print(json.dumps(pos2att, indent=2), file=fout)
# fout.close()
pos2att = json.load(open("attention_distribution.json", "r"))

layer = list(range(0, 12, 3))
data = np.array([[pos2att[i][str(dis)]["score"] / pos2att[i][str(dis)]["times"] for i in layer] for dis in range(-512, 512) if str(dis) in pos2att[0] and abs(dis) > -1]) # 512 * 2, 12
index = np.array([dis for dis in range(-512, 512) if str(dis) in pos2att[0] and abs(dis) > -1])
print(data.shape, index.shape)
wide_df = pd.DataFrame(data, index, layer)
ax = sns.lineplot(data=wide_df)
plt.savefig("attention_distribution.pdf")

