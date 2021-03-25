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
        self.id2law = {}
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        for key in json.load(open('/data/xcj/LegalBert/LegalBert/utils/glaw2num.json', 'r')):
            self.id2law[len(self.id2law)] = key
        self.init = False

    def initilze(self):
        self.init = True
        law2content = {}
        self.law2content = {}
        laws = json.load(open('/data/xcj/LegalBert/LegalBert/utils/good_laws.json', 'r'))
        for law in laws:
            year = law['date']['year'] if 'date' in law else ''
            lawkey = '%s(%s)' % (law['title'], year)
            if lawkey not in self.law2content:
                law2content[lawkey] = {} #{tiao: {'一': chapter[tiao]} if type(chapter[tiao]) == str else chapter[tiao] for chapter in law['content'] for tiao in chapter}
                self.law2content[lawkey] = []
            for chapter in law['content']:
                for tiao in chapter:
                    law2content[lawkey][tiao] = {}
                    if type(chapter[tiao]) == str:
                        self.law2content[lawkey].append(self.tokenizer.encode(chapter[tiao], add_special_tokens=False))
                        law2content[lawkey][tiao]['一'] = chapter[tiao]
                    else:
                        for kuan in chapter[tiao]:
                            self.law2content[lawkey].append(self.tokenizer.encode(chapter[tiao][kuan], add_special_tokens=False))
                            law2content[lawkey][tiao][kuan] = chapter[tiao][kuan]
            if len(self.law2content[lawkey]) == 0:
                self.law2content.pop(lawkey)
                # print(lawkey)
        self.lawnames = list(self.law2content.keys())
        self.id2arr = {}

        for lid in self.id2law:
            try:
                key = self.id2law[lid].split('-')
                content = law2content[key[0]][key[1]][key[2]]
                tokenids = self.tokenizer.encode(content, add_special_tokens=False)
                self.id2arr[lid] = tokenids
            except:
                # print(self.id2law[lid])
                pass
                # gg

    def get_law_tokens(self, lawids):
        if len(lawids) == 0:
            return [], []
        lid = random.choice(lawids)
        tokens = self.id2arr[lid]
        negative = self.sample_negative(lid)
        return tokens, negative

    def sample_negative(self, lawid):
        if random.random() < 0.3:
            return random.choice(self.law2content[self.id2law[lawid].split('-')[0]])
        else:
            return random.choice(self.law2content[random.choice(self.lawnames)])

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

'''
class MultiDocLawDataset(Dataset):
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
    
    def __getitem__(self, idx):
        ridx = int(self.idlist[idx])
        for i in range(len(self.lens)):
            if ridx >= self.lens[i]:
                ridx -= self.lens[i]
            else:
                return self.datasets[i][ridx]
        raise ValueError('Index is larger than the number of data')
    
    def __len__(self):
        return self.length
'''