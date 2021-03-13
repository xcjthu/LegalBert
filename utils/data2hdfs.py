import h5py
from transformers import AutoTokenizer
import numpy as np
import os
import json

path = "/data/disk1/private/xcj/LegalBert/data/textual_data"
savepath = "/data/disk1/private/xcj/LegalBert/data/documents.hdf5"

tokenizer = AutoTokenizer.from_pretrained("schen/longformer-chinese-base-4096")
max_len = 2048
fnames = os.listdir(path)
fnames.sort()

fout = h5py.File(savepath, "w")
dataset = fout.create_dataset("legal_document", (1000, max_len), dtype = 'int16')
dataset[:,:] = tokenizer.pad_token_id
fout.close()

fout = h5py.File(savepath, "r+")
dataset = fout["legal_document"]
docid = 0
for fn in fnames:
    fpath = os.path.join(path, fn)
    data = json.load(open(fpath, 'r'))
    for doc in data:
        ss = doc['SS'].strip()
        if len(ss) < 50:
            continue
        tokens = tokenizer(ss)['input_ids'][:max_len]
        dataset[docid,:len(tokens)] = tokens
        docid += 1
fout.close()
