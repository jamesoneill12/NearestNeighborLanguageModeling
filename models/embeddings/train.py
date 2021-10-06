import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD
from configs.models.embeddings import get_embedding_args
from utils.batchers import BatchLoader
from models.networks.generative.vae.parameters import Parameters
from models.loss.nll import NLL


def train_embeddings(data_path, emb_p=None, bsize=10, nsample=5, niters=int(1e4)):

    args = get_embedding_args(data_path, embeddings_path=emb_p, batch_size=bsize,
                              num_sample=nsample, num_iterations=niters)
    batch_loader = BatchLoader(data_path)
    params = Parameters(batch_loader.max_word_len,
                        batch_loader.max_seq_len,
                        batch_loader.words_vocab_size,
                        batch_loader.chars_vocab_size)

    neg_loss = NLL(params.word_vocab_size, params.word_embed_size)
    if args.use_cuda:
        neg_loss = neg_loss.cuda()

    # NEG_loss is defined over two embedding matrixes with shape of [params.word_vocab_size, params.word_embed_size]
    optimizer = SGD(neg_loss.parameters(), 0.1)

    for iteration in range(args.num_iterations):

        input_idx, target_idx = batch_loader.next_embedding_seq(args.batch_size)

        input = Variable(torch.from_numpy(input_idx).long())
        target = Variable(torch.from_numpy(target_idx).long())
        if args.use_cuda:
            input, target = input.cuda(), target.cuda()

        out = neg_loss(input, target, args.num_sample).mean()

        optimizer.zero_grad()
        out.backward()
        optimizer.step()

        if iteration % 500 == 0:
            out = out.cpu().data.numpy()
            if out.size != 1: out = out[0]
            print('iteration = {}, loss = {}'.format(iteration, out))

    word_embeddings = neg_loss.input_embeddings()
    np.save(args.embeddings_path, word_embeddings)


if __name__ == '__main__':

    data_path = 'C:/Users/jamesoneill/Projects/NLP/GOLM/golm/golm/golm_hil/data/ptb/'
    emb_path = data_path + 'word_embeddings.npy'
    bsize = 64
    n_iters = int(1e5)
    train_embeddings(data_path, emb_path, bsize=bsize, niters=n_iters)
