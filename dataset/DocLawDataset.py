import json
import os
from torch.utils.data import Dataset
from .IndexedDataset import make_dataset,MMapIndexedDataset
import random
import numpy as np
from transformers import AutoTokenizer
from .LawDataset import LawDataset

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
        # for i in range(sent.shape[0]-1, 0, -1):
        #     if sent[i] == 102:
        #         break
        for i in range(max(sent.shape[0] - 52, 0), sent.shape[0]):
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
