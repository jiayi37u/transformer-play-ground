import torch
from model import make_model
from prepare_data import subsequent_mask


def inference_test():
    test_model = make_model(src_vocab=11, tgt_vocab=11, N=2)
    test_model.eval()

    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])  # 输入序列[1, 10][batch, seq_len]
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)  # [1, 10, 512][batch, seq_len, embed]
    tgt = torch.zeros(1, 1).type_as(src)  # 目标序列 start token

    for i in range(9):
        # [1, x, 512][batch, cur_len, embed]当前输出序列的长度
        out = test_model.decode(tgt, memory, src_mask, subsequent_mask(tgt.size(1)).type_as(src.data))
        prob = test_model.generator(out[:, -1])  # [1, 11][batch, vocab]
        _, next_word = torch.max(prob, dim=1)  # 概率值，索引
        next_word = next_word.data[0]
        tgt = torch.cat([tgt, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1)  # 拼接预测结果
        print("Example Untrained Model Prediction:", tgt)


if __name__ == '__main__':
    inference_test()