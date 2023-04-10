import os
import torch
import numpy as np
from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable
from parser1 import args


# 长度不够的补齐
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    MaxL = max(L)
    return np.array([
        np.concatenate([x, [padding] * (MaxL - len(x))]) if len(x) < MaxL else x for x in X
    ])


# Mask out subsequent positions.
def subsequent_mask(size):
    attn_shape = (1, size, size)
    # torch.triu返回矩阵右上三角部分，其余部分为0
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    """
    i表示所有向量，j表示看到第几个向量
    for i in range(size): for j in range(size): if i >= j: mask
    tensor([[[ True, False, False],
            [ True,  True, False],
            [ True,  True,  True]]])
    """
    return subsequent_mask == 0


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):

        src = torch.from_numpy(src).to(args.device).long()
        trg = torch.from_numpy(trg).to(args.device).long()

        self.src = src  # [batch, max_len]
        self.src_mask = (src != pad).unsqueeze(-2)  # [batch, 1, max_len]
        if trg is not None:
            self.trg = trg[:, :-1]  # [batch, max_len - 1]
            self.trg_y = trg[:, 1:]
            # [batch, max_len - 1, max_len - 1] 第i行有i个true和max_len-i个false i从1开始
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class PrepareData:
    def __init__(self):

        # 读取数据 并分词 ['BOS', '你', '好', '。', 'EOS']
        self.train_en, self.train_cn = self.load_data(args.train_file)
        self.dev_en, self.dev_cn = self.load_data(args.dev_file)

        # 构建单词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)
        # print(self.cn_word_dict['我'], self.cn_index_dict[10])

        # id化
        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.wordToID(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)
        # print('wordToID', self.train_cn[:3])

        # 划分batch + padding + mask
        self.train_data = self.splitBatch(self.train_en, self.train_cn, args.batch_size)
        self.dev_data = self.splitBatch(self.dev_en, self.dev_cn, args.batch_size)
        # print('划分batch padding mask', self.train_data[0].src[:3])

    def load_data(self, path):
        en, cn = [], []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                en.append(["BOS"] + word_tokenize(line[0].lower()) + ["EOS"])
                cn.append(["BOS"] + word_tokenize(" ".join([w for w in line[1]])) + ["EOS"])
        return en, cn

    def build_dict(self, sentences, max_words=50000):
        word_count = Counter()

        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1

        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2

        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = args.UNK
        word_dict['PAD'] = args.PAD

        index_dict = {v: k for k, v in word_dict.items()}

        return word_dict, total_words, index_dict

    def wordToID(self, en, cn, en_dict, cn_dict, sort=True):
        length = len(en)
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        # sort sentences by english lengths
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 把中文和英文按照同样的顺序排序
        if sort:
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]

        return out_en_ids, out_cn_ids

    def splitBatch(self, en, cn, batch_size, shuffle=True):
        idx_list = np.arange(0, len(en), batch_size)  # 切片做batch
        if shuffle:
            np.random.shuffle(idx_list)
        batch_indexs = []  # 每个batch里的编号
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))

        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_cn))

        return batches


if __name__ == '__main__':
    data = PrepareData()

    print('seq_padding', seq_padding([[1], [2, 3], [4, 5, 6]]))
    print('wordToID', data.train_cn[:3])
    print('划分batch padding mask', data.train_data[0].trg[:2])
    print('trg_y', data.train_data[0].trg_y[:2])
    print('trg_mask', data.train_data[0].trg_mask[:1])
