import math
import numpy as np
from loaders.other import flatten, embedding_dict, softmax, normalize
from itertools import tee, islice, chain
import torch


def get_attributes(tag):
    vocab = list(tag.vocab.stoi)
    vocab2ind = {word:i for i, word in enumerate(vocab)}
    vocab_freqs = tag.vocab.freqs
    return vocab, vocab2ind, vocab_freqs


def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n-1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n-1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):

    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def bigrams(sequence, **kwargs):
    for item in ngrams(sequence, 2, **kwargs):
        yield item


def freq_dist(sent):
    bigs = list(bigrams(sent))
    uniqueBigs = dict(zip(list(set(bigs)), [0]*len(set(bigs))))
    for big in bigs:
        uniqueBigs[big] += 1
    val_sort = np.argsort(list(uniqueBigs.values()))[::-1]
    unique_bigs = [(k,v) for k, v in uniqueBigs.items()]
    unique_bigs = [unique_bigs[vs] for vs in val_sort]
    return bigs, uniqueBigs


class TransitionMatrix:
    """
    Replaces the Part of Speech Tags or Vocabulary according to transition probabilities
    with probability sp. This allows us to explore

    Arguments: ud_tag is type data.Field and train is from UDPOS.splits
    """

    def __init__(self, ud_tag, train, ptb_tag=None, devicer='cpu'):

        self.devicer = devicer
        # these are udtag corpus things
        self.vocab, self.vocab2ind, self.vocab_freqs = get_attributes(ud_tag)
        self.udtag_corpus = list(flatten(list(train.udtag)))

        # actual corpus tokens
        self.corpus = list(flatten(list(train.word)))

        self.get_co_occurrence_matrix(ptb=False)

        if ptb_tag is not None:
            self.ptb_vocab, self.ptb_vocab2ind, self.ptb_vocab_freqs = get_attributes(ptb_tag)
            self.ptb_corpus = list(flatten(list(train.ptb_tag)))
            self.get_co_occurrence_matrix(ptb=True)

        self.temp = 1.0

    def get_uni_probs(self):
        ind2prob = {}
        for word in self.vocab.keys():
            for tag in self.vocab_freqs.keys():
                if word == tag:
                    ind2prob[self.vocab[word]] = float(self.vocab_freqs[tag])
        x = np.array(list(ind2prob.values()))
        x = softmax(normalize(x))
        ind2prob = dict(zip(ind2prob.keys(), x))
        return ind2prob

    def ngrams(self, lst, n):
        tlst = lst
        while True:
            a, b = tee(tlst)
            l = tuple(islice(a, n))
            if len(l) == n:
                yield l
                next(b)
                tlst = b
            else:
                break

    def get_co_occurrence_matrix(self, vocab=None, ptb=False):
        if ptb:
            corpus = self.ptb_corpus
            vocab2ind = self.ptb_vocab2ind
            if vocab is None:
                vocab = self.ptb_vocab
        else:
            corpus = self.udtag_corpus
            vocab2ind = self.vocab2ind
            if vocab is None:
                vocab = self.vocab

        bi_grams, bigram_freq = freq_dist(corpus)
        co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))

        for bigram, freq in bigram_freq.items():
            current = bigram[1]
            previous = bigram[0]
            count = freq
            pos_current = vocab2ind[current]
            pos_previous = vocab2ind[previous]
            co_occurrence_matrix[pos_current][pos_previous] = count
        co_occurrence_matrix = np.matrix(co_occurrence_matrix).astype(float)
        co_occurrence_matrix = softmax(np.nan_to_num(normalize(co_occurrence_matrix)))

        if ptb:
            self.ptb_emission = co_occurrence_matrix
        else:
            self.emission = co_occurrence_matrix

    def sample_neighbor_sequence(self, batch_ids, sp, ptb=False):
        emission = self.ptb_emission if ptb else self.emission
        if type(batch_ids) is torch.Tensor and sp != 0.0:
            batch_ids = batch_ids.cpu().data.numpy()
            batch_shape = batch_ids.shape
            batch_ids = batch_ids.flatten()
            bools = np.random.choice(2, size=len(batch_ids), p=[1.0 - sp, sp])
            inds = np.where(bools)
            sampled_s = batch_ids[inds]
            samp_indices = []
            for ind in sampled_s:
                em_row = np.asarray(emission[:, ind])
                tp = np.squeeze(em_row)
                assert emission.shape[0] == len(tp)
                samp_ind = np.random.choice(emission.shape[0], 1, p=tp)
                samp_indices.append(int(samp_ind))
            batch_ids[inds] = np.squeeze(samp_indices)
            batch_ids = batch_ids.reshape(batch_shape)
            x_batch = torch.from_numpy(batch_ids).type(torch.LongTensor).to(self.devicer)
            return x_batch

