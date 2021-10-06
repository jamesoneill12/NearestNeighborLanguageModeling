import torch
import numpy as np
import data


def word_count(corpus):
    counter = [0] * len(corpus.dictionary.idx2word)
    for i in corpus.train:
        counter[i] += 1
    for i in corpus.valid:
        counter[i] += 1
    for i in corpus.test:
        counter[i] += 1
    return np.array(counter).astype(int)


def word_freq_ordered(corpus):
    # Given a word_freq_rank, we could find the word_idx of that word in the corpus
    counter = word_count(corpus = corpus)
    # idx_order: freq from large to small (from left to right)
    idx_order = np.argsort(-counter)
    return idx_order.astype(int)


def word_rank_dictionary(corpus):
    # Given a word_idx, we could find the frequency rank
    # (0-N, the smaller the rank, the higher frequency the word) of that word in the corpus
    idx_order = word_freq_ordered(corpus = corpus)
    # Reverse
    rank_dictionary = np.zeros(len(idx_order))
    for rank, word_idx in enumerate(idx_order):
        rank_dictionary[word_idx] = rank
    return rank_dictionary.astype(int)


class Rand_Idxed_Corpus(object):
    # Corpus using word rank as index
    def __init__(self, corpus, word_rank):
        self.dictionary = self.convert_dictionary(dictionary = corpus.dictionary, word_rank = word_rank)
        self.train = self.convert_tokens(tokens = corpus.train, word_rank = word_rank)
        self.valid = self.convert_tokens(tokens = corpus.valid, word_rank = word_rank)
        self.test = self.convert_tokens(tokens = corpus.test, word_rank = word_rank)

    def convert_tokens(self, tokens, word_rank):
        rank_tokens = torch.LongTensor(len(tokens))
        for i in range(len(tokens)):
            #print(word_rank[tokens[i]])

            rank_tokens[i] = int(word_rank[tokens[i]])
        return rank_tokens

    def convert_dictionary(self, dictionary, word_rank):
        rank_dictionary = data.Dictionary()
        rank_dictionary.idx2word = [''] * len(dictionary.idx2word)
        for idx, word in enumerate(dictionary.idx2word):
            #print(word_rank)
            #print(rank)
            rank = word_rank[idx]
            rank_dictionary.idx2word[rank] = word
            if word not in rank_dictionary.word2idx:
                rank_dictionary.word2idx[word] = rank
        return rank_dictionary