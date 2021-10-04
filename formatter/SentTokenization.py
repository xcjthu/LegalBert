import torch
import random
from transformers import AutoTokenizer

class SentTokenization(object):
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.sent_tokenids = list(range(1, 100)) + list(range(261, 405)) # some unused token, including the official unused tokens and some tokens which seems that they will never be used
        # self.punctuations = set([8024, 511, 8043, 8013, 8039]) # 包括 ，。？！  这里改了的话一定记得去改模型里面给sentence id的部分
        self.punctuations = set([511, 8043, 8013, 8039]) # 包括 ，。？！  这里改了的话一定记得去改模型里面给sentence id的部分

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
    
    def split_sents(self, doc):
        sents = []
        last_pos = 1

        for tpos, token in enumerate(doc[1:-1]):
            if token in self.punctuations:
                sents.append(doc[last_pos:tpos + 2].tolist())
                last_pos = tpos + 2

        sents.append(doc[last_pos:-1].tolist())
        return [s for s in sents if len(s) > 0]

    def split_for_recurrent(self, docs, block_len=512, pad_id=0, max_block_size=8):
        selected = [docs[0][0]]
        blocks, now_inp_block = [], []
        mask, smask = [], []
        for doc in docs:
            now_inp_block.append(doc[0])
            sents = self.split_sents(doc)
            mask_sent_num = max(1, int(len(sents) * 0.15)) if len(sents) > 3 else 0 # select some sents, to predict the masked token (the most token will be masked)
            mask_sent_id = set(random.sample(list(range(len(sents))), mask_sent_num))
            for i in range(len(sents)):
                if len(blocks) >= max_block_size:
                    break
                if i in mask_sent_id and len(selected) < block_len: # we will only feed the selected sent to one block, therefore, it can not be too long
                    selected += sents[i]
                    selected.append(doc[-1])
                else:
                    if len(sents[i]) > block_len:
                        sents[i] = sents[i][:block_len]
                    if block_len - len(now_inp_block) < len(sents[i]):
                        mask.append(([1] * len(now_inp_block) + [0] * (block_len - len(now_inp_block)))[:block_len])
                        now_inp_block += [pad_id] * (block_len - len(now_inp_block))
                        blocks.append(now_inp_block[:block_len])
                        now_inp_block = []
                    now_inp_block += sents[i]
            if len(now_inp_block) > 0:
                now_inp_block.append(doc[-1])
        if len(now_inp_block) > 0 and len(blocks) < max_block_size:
            mask.append(([1] * len(now_inp_block) + [0] * (block_len - len(now_inp_block)))[:block_len])
            now_inp_block += [pad_id] * (block_len - len(now_inp_block))
            blocks.append(now_inp_block[:block_len])

        for b in blocks:
            assert len(b) == block_len
        if len(blocks) < max_block_size:
            mask += [[0] * block_len] * (max_block_size - len(blocks))
            blocks += [[pad_id] * block_len] * (max_block_size - len(blocks))
            
        assert len(blocks) == max_block_size
        if len(selected) > block_len:
            selected = selected[:block_len]
            smask = [1] * block_len
        else:
            smask = [1] * len(selected) + [0] * (block_len - len(selected))
            selected += [pad_id] * (block_len - len(selected))
        return blocks, selected, mask, smask

