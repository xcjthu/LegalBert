import json
from tqdm import tqdm
import os

all_law = {}
path = '/mnt/datadisk0/xcj/LegalBert/data/formatted_laws'
for fname in ['formatted_fl.json', 'formatted_sfjs.json', 'formatted_xzfg.json']:
    laws = json.load(open(os.path.join(path, fname), 'r'))
    for law in laws:
        all_law[law['id_']] = law

path = '/mnt/datadisk0/xcj/LegalBert/data/'
used_law = {}
for p in ['ms_plm', 'xs_plm']:
    fnames = os.listdir(os.path.join(path, p))
    for fn in tqdm(fnames):
        data = json.load(open(os.path.join(path, p, fn)))
        for doc in data:
            for law in doc['term_ids']:
                key = '_'.join([str(l) for l in law])
                if not key in used_law:
                    used_law[key] = 0
                used_law[key] += 1

fout = open('used_law_num.json', 'w')
print(json.dumps(used_law, ensure_ascii = False, indent = 2), file = fout)
fout.close()


