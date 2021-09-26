from sklearn import cluster
import torch
import json
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from transformers import BertModel
import numpy as np
from torch import nn
from sklearn.cluster import *
import time
data = [
    "The son of former NBA player Dell and older brother of current NBA player Seth, Curry played college basketball for the Davidson Wildcats. Curry won his first MVP award and led the Warriors to their first championship since 1975. Curry was selected with the seventh overall pick in the 2009 NBA draft by the Golden State Warriors.",
    "Born and raised in Queens, New York City, Trump graduated from the University of Pennsylvania. Previous neural coherence models have focused on identifying semantic relations between adjacent sentences. First, a topic model is constructed under the so-called sparsity assumption which states that words in a segment should have the same small set of topics.",
]

gpu = 4
layernum = 12
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name).to(gpu)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
token2sim = {}

cosine = nn.CosineSimilarity(dim=-1)
for doc in tqdm(data):
    inputs = tokenizer(doc, return_tensors="pt", max_length=512, truncation=True)
    for key in inputs:
        inputs[key] = inputs[key].to(gpu)
    outputs = model(**inputs)
    final_layer = outputs["last_hidden_state"][0] # FloatTensor: batch, seq_len, hidden_size
    cos_dis = cosine(final_layer.unsqueeze(1), final_layer.unsqueeze(0))
    hiddens = np.array(final_layer.tolist())
    cluster_model = KMeans(n_clusters=8, random_state=0).fit(hiddens)
    begin = time.time()
    # cluster_model = DBSCAN(eps=0.8, n_jobs=10).fit(cos_dis.tolist())
    # cluster_model = AgglomerativeClustering(n_clusters=5, affinity="precomputed", linkage="average").fit(cos_dis.tolist())
    end = time.time()
    print("time", end - begin)
    labels = cluster_model.labels_.tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(labels)
    print(list(zip(tokens, labels)))
    print("==" * 20)



