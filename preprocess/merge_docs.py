import json
import os
from tqdm import tqdm

path = "/data/disk1/private/xcj/LegalBert/data/textual_data/xs"
fnames = os.listdir(path)
fout = open(path + ".txt", "w")
for fn in tqdm(fnames):
    fin = open(os.path.join(path, fn), "r")
    docs = json.load(fin)
    for doc in docs:
        print(json.dumps(doc, ensure_ascii=False), file=fout)
fout.close()

