import json
import os
from tqdm import tqdm
title = set()
for folder in ['ms_plm', 'xs_plm']:
    fnames = os.listdir(os.path.join('/data/xcj/LegalBert/data/', folder))
    for fn in tqdm(fnames):
        data = json.load(open(os.path.join('/data/xcj/LegalBert/data/', folder, fn), 'r'))
        for doc in data:
            if len(doc['SS']) < 50:
                continue
            for law in doc['related_laws']:
                title.add(law)
fout = open('related_laws.txt', 'w')
print(json.dumps(list(title), ensure_ascii = False, indent = 2), file = fout)
fout.close()

