import torch
import random
from transformers import AutoTokenizer

class SentTokenization(object):
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.sent_tokenids = list(range(1, 100)) + list(range(261, 405)) # some unused token, including the official unused tokens and some tokens which seems that they will never be used
        self.punctuations = set([8024, 511, 8043, 8013, 8039]) # 包括 ，。？！  这里改了的话一定记得去改模型里面给sentence id的部分

    def shuffle_part_doc(self, doc, shuffle_ratio):
        # doc: doc_len (list)
        sents = []
        last_pos = 1
        for tpos, token in enumerate(doc[1:-1]):
            if token == 511:
                sents.append(doc[last_pos:tpos + 2])
                last_pos = tpos + 2
        sents.append(doc[last_pos:-1])
        # if len(sents) < 3:
        #     return doc.tolist()
        selected = random.sample(list(range(len(sents))), int(len(sents) * shuffle_ratio))
        shuffle = {i: i for i in range(len(sents))}
        for i in range(-1, len(selected) - 1):
            shuffle[selected[i]] = selected[i + 1]
        ret = [int(doc[0])]
        for sid, sent in enumerate(sents):
            ret += sents[shuffle[sid]].tolist()
        # print(self.tokenizer.decode(ret))
        # print(self.tokenizer.decode(doc))
        # print("==" * 20)
        ret.append(int(doc[-1]))
        if len(ret) != len(doc):
            print("shuffle:", shuffle)
            print("selected:", selected)
            print("sents:", sents)
            print("docs:", doc)
        assert len(ret) == len(doc)
        return ret

    def add_sent_token(self, doc, prefix_len=0):
        sents = []
        last_pos = 1
        
        
        for tpos, token in enumerate(doc[1:-1]):
            if token in self.punctuations:
                sents.append(doc[last_pos:tpos + 2])
                last_pos = tpos + 2

        sents.append(doc[last_pos:-1])
        ret = [int(doc[0])]
        sent_pos = []
        nowsid, sentids = 3, [1] # 1 for cls, 2 for sep, 0 for pad
        for sid, sent in enumerate(sents):
            sent_pos.append(len(ret) + prefix_len)
            ret.append(self.sent_tokenids[sid])
            ret += sent.tolist()
            sentids += [nowsid] * (len(sent) + 1)
            nowsid += 1
        ret.append(int(doc[-1]))
        sentids.append(2)
        assert len(ret) == len(doc) + len(sents) == len(sentids)
        return ret, sent_pos, sentids
