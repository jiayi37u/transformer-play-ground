import torch
import torch.nn as nn
import torch.nn.functional as F

from parser1 import args
from prepare_data import PrepareData
from model import make_model
from utils import LabelSmoothing, NoamOpt, SimpleLossCompute
from train import train


def main():
    # 数据预处理
    data = PrepareData()
    args.src_vocab = len(data.en_word_dict)
    args.tgt_vocab = len(data.cn_word_dict)
    print("src_vocab %d" % args.src_vocab)
    print("tgt_vocab %d" % args.tgt_vocab)

    # 初始化模型
    model = make_model(src_vocab=args.src_vocab,
                       tgt_vocab=args.tgt_vocab,
                       N=args.layers,
                       d_model=args.d_model,
                       d_ff=args.d_ff,
                       h=args.h_num,
                       dropout=args.dropout)

    if args.type == 'train':
        print('####### start train #######')
        criter = LabelSmoothing(args.tgt_vocab, padding_idx=0, smoothing=0.0)
        optimizer = NoamOpt(args.d_model, factor=1, warmup=2000,
                            optimizer=torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9))
        train(data, model, criter, optimizer)


if __name__ == '__main__':
    main()