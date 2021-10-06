import os
import torch
import numpy as np
from loaders.misc import embedding_dict, clean_str
from loaders.codebook import Codebook
from loaders.helpers import save_wiki102_vocab
from util.neighbor import get_nearest_vocab_neighbor
from macros import *
from loaders.dictionary import Dictionary, PretrainedDictionary
from util.bpe import BytePairEncoder
import time


def wikitext102(pretrain=False, emb_type = 'word'):
    corpus = Corpus(WIKITEXT2_ROOT, emb_type='word', pretrain=pretrain)
    corp = {}
    corp['word2ind'] = corpus.dictionary.word2idx
    print("Vocabulary Size: {}".format(len(corp['word2ind'])))
    corp['ind2word'] = corpus.dictionary.idx2word
    corp['word2vec'] = corpus.dictionary.wv
    corp['id2vec'] = None
    return corp


def build_unigram_noise(freq):
    """build the unigram noise from a list of frequency
    Parameters:
        freq: a tensor of #occurrences of the corresponding index
    Return:
        unigram_noise: a torch.Tensor with size ntokens,
        elements indicate the probability distribution
    """
    total = freq.sum()
    noise = freq / total
    assert abs(noise.sum() - 1) < 0.001
    return noise


class Corpus(object):
    def __init__(self, path, emb_type=None, clean=False, limit=None,
                 pretrain=False, ns=False, error_coding=False, count=False,
                 reward=False, lcs=False, neg_samps=False, bpe=False):
        """

        :param path: path to dataset
        :param emb_type: if pretrained embeddings are used, choose which ones
        :param clean: whether to preprocess the dataset
        :param limit: caps the number of tokens in vocab by frequency
        :param pretrain: if true, use pretrained emebeddings
        :param ns: if true, use neighborhood sampling via cosine similarities of pretrained embs
        :param error_coding: if true, generate codebook for training vocab
        :param count: if true, get dictionary counts for each token (used for nce)
        :param reward: if not false, load of pretrained embeddings (needed to compute rewards)
        :param lcs: if true and limit true, we assign terms over the limit to
                    idx inside vocab via longest common subsequence
        :param neg_samps: when using multiclass binary classification where we
                        predict the target and its N neighbors we also need to retrieve
                        negative samples which are outside N neighbors. In the BinaryCrossEntropy loss
                        we can pass a unigram distribution to draw negative samples non-uniformly.
        """

        self.dictionary = Dictionary()
        self.limit = limit
        self.lcs = lcs
        self.neg_samps = neg_samps
        self.error_coding = error_coding
        self.count = count
        if self.count:
            self.neg_samps = count

        endext = '.tokens' if '3' in path else '.txt'
        startext = 'wiki.' if '3' in path else ''

        train_path = os.path.join(path, startext + 'train' + endext)
        self.valid_path = os.path.join(path, startext + 'valid' + endext)
        self.test_path = os.path.join(path, startext + 'test' + endext)

        tokenizer = self.tokenize if endext == '.tokens' else self.tokenize

        print("Limit is {}".format(limit))
        if self.limit:
            self.limit_vocab_created = False
            if type(self.limit) != int:
                limit = int(self.limit)
            self.token_count = {}
        if self.count:
            self.token_count = {}

        self.train = tokenizer(train_path, limit=self.limit,
                               clean=clean, lcs=lcs, count=self.neg_samps)

        if self.limit: self.limit_vocab_created = True
        self.valid = tokenizer(self.valid_path)
        self.test = tokenizer(self.test_path)

        if self.limit:
            """needed so that when getting ntokens from __len__"""
            # self.dictionary.word2idx = self.limword2idx
            print("Size of limited vocab {}".format(len(self.dictionary)))

        # not needed after processing
        if hasattr(self.dictionary, 'embeddings'):
            del self.dictionary.embeddings

        # we need to do this for nsampling, sampling neighbors at training time is not reliable.
        if ns and pretrain is False:
            self.dictionary.wv = embedding_dict(self.dictionary.word2idx)
            print("{} words in vocab".format(len(self.dictionary.word2idx)))
            self.neighbourhood_dict = self.get_neighbor_dict()

        if reward:
            """the reward loss expects a tensor to lookup"""
            self.dictionary.wv = embedding_dict(self.dictionary.word2idx)
            self.dictionary.wv = torch.from_numpy(
                np.array(list(self.dictionary.wv.values()))
            ).type(torch.cuda.FloatTensor)

        if type(self.dictionary) == PretrainedDictionary:
            self.neighbourhood_dict = self.get_neighbor_dict()

        if error_coding:
            """Error codes are random for now"""
            word_range = list(range(len(self.dictionary)))
            self.codebook = Codebook()
            self.error_codes = self.codebook.get_binary_code(word_range)
        else:
            self.error_codes = None

    # creates a dictionary with values of neighbor array
    # (2, len(vocab)) = 1st column is neighbor id and 2nd column is sample
    def get_neighbor_dict(self):
        # this not all embeddings, just that from the vocab
        neighbor_lookup = get_nearest_vocab_neighbor(self.dictionary.word2idx, self.dictionary.wv)
        return neighbor_lookup

    def sample_neighbor(self, word_id):
        ids, id_probs = self.neighbourhood_dict[word_id]
        # samples a top neighor
        # AH ! THIS IS THE <EOS> TAG SO IGNORE
        if float('nan') in id_probs:
            return word_id
        chosen_neighbor_id = np.random.choice(ids, p=id_probs)
        return chosen_neighbor_id

    # replaces sequences with neighbor ids with sample probability
    # this prob should either be static, dynamic (exp decay, linear decay or sig decay)
    def sample_neighbor_sequence(self, word_ids, sample_prob):
        if type(word_ids) is torch.Tensor and sample_prob is not 0.0:
            word_cpu_ids = word_ids.cpu().data.numpy()
            bools = np.random.choice([0, 1], size=len(word_cpu_ids), p=[1.0-sample_prob, sample_prob])
            inds = np.where(bools)
            if bools.size != 0:
                sampled_word_ids = word_cpu_ids[inds]
                # might be faster to use above
                sample_neighbor_ids = [self.sample_neighbor(sword_id) for sword_id in sampled_word_ids]
                # now substitute sampled neighbor for original id
                word_cpu_ids[inds] = sample_neighbor_ids
                # and convert back to float tensor AND CUDA IF NECESSARY
                replaced_ids = torch.LongTensor(word_cpu_ids).cuda()
            else:
                replaced_ids = word_ids
        elif type(word_ids) is list and sample_prob is not 0.0:
            # not finished
            chosen_neighbor_ids = \
                torch.LongTensor([self.sample_neighbor(word_id) for word_id in word_ids]).cuda()
            replaced_ids = word_ids
        else:
            return word_ids
        return replaced_ids

    def sample_target_sequence(self, word_ids, pred_ids, sample_prob):
        word_ids = word_ids.clone().data.numpy()
        word_ids = word_ids.clone().data.numpy()
        bools = np.random.choice([0, 1], size=(len(word_ids),), p=[1.0 - sample_prob, sample_prob])
        inds = np.where(bools).astype(int)
        word_ids[inds] = pred_ids[inds]
        replaced_ids = torch.FloatTensor(word_ids)
        return replaced_ids

    def get_limited_vocab(self, token_count, limit):
        # just delete words which have a count > limit
        counts = np.array(list(token_count.values()), dtype=int)
        words = np.array(list(token_count.keys()))
        rare_count_inds = np.argsort(counts)[::-1][limit:]
        common_count_inds = np.argsort(counts)[::-1][:limit]

        limited_vocab = words[common_count_inds]
        oov_words = list(words[rare_count_inds])
        limit_vocab_perc = (len(oov_words) / len(token_count)) * 100

        print("Limited vocab is size {}".format(len(limited_vocab)))
        print("OOV vocab is size {}".format(len(oov_words)))

        print("{:.2f}% of OOV words for replacement".format(limit_vocab_perc))
        if self.lcs:
            print("now replacing rare words using"
                  " closest longest common subsequence ...")
        else:
            print("now replacing rare words using <unk> token")

        """creates a limited vocabulary that is needed to assign rare idx in next step."""
        for lim_word in list(limited_vocab):
            self.dictionary.limited_word(lim_word)
        return oov_words, limited_vocab

    def lcs_vocab_limit(self, token_count, limit):
        oov_words, _ = self.get_limited_vocab(token_count, limit)
        start = time.time()
        for i, rword in enumerate(oov_words):
            """better to replace word than remove so ids can be assigned down below"""
            # self.dictionary.remove_word(rword)
            if i % 100 == 0:
                end = time.time()
                print("{} words processed for replacement, it took {}secs".format(i, start - end))
                start = time.time()
            self.dictionary.replace_word(rword, normalize=True)
        print("replacement finished ! it took {} seconds".format(end - start))

    def tokenize_file(self, path, tokens, clean):
        # Tokenize file content
        with open(path, mode='r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                if clean:
                    line = clean_str(line)
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids

    def tokenize_limit_file(self, path, tokens, limited_vocab=None):
        """
        When
        :param path:
        :param tokens:
        :param limited_vocab:
        :return:
        """

        # Tokenize file content and assign id from limited vocab
        if self.limit_vocab_created is False:
            # need to convert all original indices to limited idx
            self.idx2limidx = {self.dictionary.word2idx[word]: i for (i, word) in enumerate(limited_vocab)}
            temp_dict = {}
            for word in list(self.dictionary.word2idx.keys()):
                if word in limited_vocab:
                    temp_dict[word] = self.idx2limidx[self.dictionary.word2idx[word]]
                else:
                    temp_dict[word] = self.dictionary.word2idx['<unk>']

            self.dictionary.word2idx = temp_dict
            del temp_dict

            """
            self.limword2idx = {word : self.idx2limidx[self.dictionary.word2idx[word]]
                        for word in limited_vocab}
            print("Limited vocab is of size {}".format(len(self.limword2idx)))
            """

        with open(path, mode='r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
                    """
                    if word in self.limword2idx:
                        ids[token] = self.limword2idx[word]
                    else:
                        ids[token] = self.limword2idx['<unk>']
                    token += 1                    

                    """
        return ids

    def tokenize(self, path, limit=None,
                 clean=False, lcs=False, count=False):

        """
        Tokenizes a text file

        :param
        clean: cleans string of any unneeded punctuation
        limit: when not None and LCS is False,
               just assign oov words to <unk>.
        lcs: when true and limit not None, we assign oov
                words to nearest in-vocab word according to minimum lcs distance (this is slow atm)
        """

        path = ROOT_PATH + path
        assert os.path.exists(path)
        # Add words to the dictionary

        with open(path, mode='r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                if clean:
                    line = clean_str(line)
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    if limit is not None or count:
                        if word in self.token_count:
                            self.token_count[word] += 1
                        else:
                            self.token_count[word] = 1
                    self.dictionary.add_word(word)

        if limit is not None:
            """  has to be next condition because token_count
            doesn't exist unless limit is not None """
            if len(self.token_count) > limit:
                if lcs:
                    self.lcs_vocab_limit(self.token_count, limit)
                    ids = self.tokenize_file(path, tokens, clean)
                else:
                    _, limited_vocab = self.get_limited_vocab(self.token_count, limit)
                    print("Size of limite")
                    ids = self.tokenize_limit_file(path, tokens, limited_vocab)
            else:
                ids = self.tokenize_file(path, tokens, clean)
        elif self.limit:
            print("Now computing valid and test")
            # no need for limited vocab now since limword2id already created
            ids = self.tokenize_limit_file(path, tokens)
            if 'val' in path:
                self.val_ids = ids
            elif 'test' in path:
                self.test_ids = ids
        else:
            ids = self.tokenize_file(path, tokens, clean)

        if hasattr(self.dictionary, 'vocab_perc'):
            vocab_perc = round((1 - self.dictionary.vocab_perc / len(ids)) * 100, 2)
            print("{} % in the pretrained vocab.".format(vocab_perc))

        """get counts from corpus if negative sampling is true"""
        if count: self.id_counts = ids
        return ids


if __name__ == "__main__":

    vocab_attrib = wikitext102(pretrain=True)
    save_wiki102_vocab(vocab_attrib)

    # sentences = wikitext103()
    # vocab = get_vocab(sentences)


