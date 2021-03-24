import json
import os
from tqdm import tqdm

lpath = '/data/xcj/LegalBert/data/law/formatted_laws'
laws = []
for fn in ['formatted_fl.json', 'formatted_sfjs.json', 'formatted_xzfg.json']:
    laws += json.load(open(os.path.join(lpath, fn), 'r'))
print('length of laws:', len(laws))
related_laws = set(json.load(open('related_laws.txt', 'r')))
laws = [law for law in laws if law['title'] in related_laws]
print('length of laws:', len(laws))

fout = open('good_laws.json', 'w')
print(json.dumps(laws, ensure_ascii=False, indent=2), file=fout)
fout.close()

law2id = {}
lid = 0
knum = 0
for law in laws:
    for chapter in law['content']:
        for tiao in chapter:
            if len(chapter[tiao]) > 1:
                knum += 1
            for kuan in chapter[tiao]:
                key = '-'.join((law['title'], tiao, kuan))
                law2id[key] = lid
                lid += 1
fout = open('law2id.json', 'w')
print(json.dumps(law2id, ensure_ascii=False, indent=2), file=fout)
fout.close()
print(len(law2id), knum)

glaw2num = dict()
for folder in ['ms_plm', 'xs_plm']:
    fnames = os.listdir(os.path.join('/data/xcj/LegalBert/data/', folder))
    for fn in tqdm(fnames):
        data = json.load(open(os.path.join('/data/xcj/LegalBert/data/', folder, fn), 'r'))
        for doc in data:
            if len(doc['SS']) < 50:
                continue
            for law in doc['related_laws']:
                for spectiao in doc['related_laws'][law]:
                    tiao = spectiao['term']
                    kuan = spectiao['kuan'] if spectiao['kuan'] != -100 else '一'
                    key = '-'.join((law, tiao, kuan))
                    if key in law2id:
                        if key not in glaw2num:
                            glaw2num[key] = 0
                        glaw2num[key] += 1
fout = open('glaw2num.json', 'w')
print(json.dumps(glaw2num, ensure_ascii=False, indent=2), file=fout)
fout.close()

