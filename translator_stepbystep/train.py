import time
import torch
from parser1 import args
from utils import SimpleLossCompute


def run_epoch(data, model, loss_func, epoch):
    start = time.time()
    total_tokens, tokens = 0, 0
    total_loss = 0
    n_accum = 0

    for i, batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_func(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (
            epoch, i - 1, loss / batch.ntokens, tokens / elapsed / 1000))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def train(data, model, criter, optimizer):
    for epoch in range(args.epochs):
        model.train()
        run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criter, optimizer), epoch)

        model.eval()
        loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criter, None), epoch)
        print('<<<<< Evaluate loss: %f' % loss)

    torch.save(model.state_dict(), args.save_file)