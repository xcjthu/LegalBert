import json
import os
from torch.utils.data import Dataset
from .IndexedDataset import make_dataset,MMapIndexedDataset
import random
import numpy as np
from transformers import AutoTokenizer

class LawDataset:
    def __init__(self, config):
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        law_id = json.load(open('/mnt/datadisk0/xcj/LegalBert/LegalBert/utils/law_id.json', 'r'))
        self.id2law = law_id['id2law']
        self.law2id = law_id['law2id']
        self.all_law = {}
        self.all_law_keys = {}
        path = '/mnt/datadisk0/xcj/LegalBert/data/formatted_laws'
        for fname in ['formatted_fl.json', 'formatted_sfjs.json', 'formatted_xzfg.json']:
            laws = json.load(open(os.path.join(path, fname), 'r'))
            for law in laws:
                content = {}
                for chapter in law['content']:
                    content.update(chapter)
                if len(content) == 0:
                    continue
                self.all_law_keys[str(law['id_'])] = []
                for tiao in content:
                    if type(content[tiao]) == str:
                        content[tiao] = {'一': content[tiao]}
                    for kuan in content[tiao]:
                        self.all_law_keys[str(law['id_'])].append('%s_%s_%s' % (str(law['id_']), tiao, kuan))
                self.all_law[str(law['id_'])] = content

        self.all_law_names = list(self.all_law_keys.keys()) # 法律
    
    def get_content_token(self, lawid):
        if type(lawid) == str:
            key = lawid
        else:
            key = self.id2law[str(lawid)]
        lid, tiao, kuan = key.split('_')
        law = self.all_law[lid]
        if tiao not in law:
            print(key)
            return None
        if kuan == '-100':
            target = ''.join([v.strip() for v in law[tiao].values()])
        else:
            if kuan not in law[tiao]:
                return None
            target = law[tiao][kuan].strip()
        return self.tokenizer.encode(target, add_special_tokens=False)

    def sample_negative(self, lawids):
        hard =  random.random() < 0.3
        exist_laws = set([self.id2law[str(lid)] for lid in lawids])
        lawnames = [l.split('_')[0] for l in exist_laws]
        if hard and len(lawids) > 0:
            select_law = random.choice(lawnames)
        else:
            select_law = random.choice(self.all_law_names)
            while select_law in lawnames:
                select_law = random.choice(self.all_law_names)
        target = random.choice(self.all_law_keys[select_law])
        while target in exist_laws:
            target = random.choice(self.all_law_keys[select_law])
        return self.get_content_token(target)


class NSPDocLawDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint('train', 'max_len')
        path = config.get('data', '%s_data' % mode)
        flist = config.get('data', '%s_files' % mode).split(',')
        self.datasets = [MMapIndexedDataset(os.path.join(path, f), False) for f in flist]
        self.lens = [len(d) for d in self.datasets]
        self.length = sum(self.lens)

        self.idlist = np.arange(0, self.length)
        np.random.shuffle(self.idlist)

        #self.indexed_dataset = make_dataset(config, mode)
        #self.data_num = len(self.indexed_dataset)
        self.lawdata = LawDataset(config)
        print(mode, 'the num of data: ', self.length)

    def __getitem__(self, idx):
        ridx = int(self.idlist[idx])
        sent = None
        for i in range(len(self.lens)):
            if ridx >= self.lens[i]:
                ridx -= self.lens[i]
            else:
                sent = self.datasets[i][ridx]
        if sent is None:
            raise ValueError('Index is larger than the number of data')

        # sent = self.indexed_dataset[idx]
        for i in range(sent.shape[0]-1, 0, -1):
            if sent[i] == 102:
                break
        if i == sent.shape[0] - 1:
            laws = []
        else:
            laws = sent[i+1:].tolist()

        if random.random() < 0.5 and len(laws) > 0:
            cand = self.lawdata.get_content_token(random.choice(laws))
            label = 1
        else:
            cand = self.lawdata.sample_negative(laws)
            label = 0

        return {
            'doc': sent[:i+1],
            'cand': cand,
            'label': label,
        }

    def __len__(self):
        return self.length
        # return len(self.indexed_dataset)


class DocLawDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint('train', 'max_len')
        path = config.get('data', '%s_data' % mode)
        flist = config.get('data', '%s_files' % mode).split(',')
        self.datasets = [MMapIndexedDataset(os.path.join(path, f), False) for f in flist]
        self.lens = [len(d) for d in self.datasets]
        self.length = sum(self.lens)
        self.idlist = np.arange(0, self.length)
        np.random.shuffle(self.idlist)

        #self.indexed_dataset = make_dataset(config, mode)
        #self.data_num = len(self.indexed_dataset)

        self.lawdata = LawDataset(config)

    def __getitem__(self, idx):
        if not self.lawdata.init:
            self.lawdata.initilze()
        ridx = int(self.idlist[idx])
        sent = None
        for i in range(len(self.lens)):
            if ridx >= self.lens[i]:
                ridx -= self.lens[i]
            else:
                sent = self.datasets[i][ridx]
        if sent is None:
            raise ValueError('Index is larger than the number of data')

        # sent = self.indexed_dataset[idx]
        for i in range(sent.shape[0]-1, 0, -1):
            if sent[i] == 102:
                break
        if i == sent.shape[0] - 1:
            laws = []
        else:
            laws = sent[i+1:].tolist()
        gt, neg = self.lawdata.get_law_tokens(laws)
        return {
            'doc': sent[:i+1],
            'gtlaw': gt,
            'neglaw': neg,
        }

    def __len__(self):
        return self.length
        # return len(self.indexed_dataset)
