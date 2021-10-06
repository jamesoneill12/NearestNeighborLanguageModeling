from loaders.helpers import get_embeddings, get_vocab_embeddings
import nltk
from collections import Counter
from loaders.misc import clean_str
from itertools import zip_longest
from loaders.strings import levenshteinDistance
from edlib import align
from pycocotools.coco import COCO
import numpy as np
import pickle
import argparse


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx2count = {}
        self.word2count = Counter()
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def create_counter(self, counts, thresh):
        """Counter with words-freq, remove ones that have freq below thresh"""
        for k in list(counts):
            if counts[k] < thresh:
                del counts[k]
        self.word2count = counts
        for (word, count) in counts.items():
            self.idx2word[self.word2idx[word]] = count
        # dict(zip(list(self.word2idx.keys())), list(counts.values()))

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def id2sent(vocab, ids, tag_split='<end>'):
    # Convert word_ids to words
    sampled_caption = []
    for word_id in ids:
        word = vocab.idx2word[int(word_id)]
        # if word not in ['<p>', '<', 'unk', '>', '</p>']:
        #    if '</p>' in word:
        # print(word)
        sampled_caption.append(word)
        if word == tag_split:
            break
    # no need to keep padding when using infersent or skipthought to compare sentence reps
    sentence = ' '.join(sampled_caption).replace("<pad>", "")
    return sentence


def ids2sents(vocab, ids, show=False):
    # Convert word_ids to words
    sampled_caption = []
    sampled_captions = []
    for word_ids in ids:
        for word_id in word_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                sentence = ' '.join(sampled_caption)
                if show: print(sentence)
                sampled_caption = []
                sampled_captions.append(sentence)
    return sampled_captions


def reorder_sents(a):
    a = [s.split() for s in a]
    a = [" ".join([x for x in s if x is not None]) for s in list(map(list, zip_longest(*a)))]
    return a


# only to be used for coco, flickr has its own one
def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


class MTDictionary(object):
    def __init__(self):
        self.en_word2idx = {}
        self.en_idx2word = []
        self.de_word2idx = {}
        self.de_idx2word = []

    def __len__(self):
        return len(self.de_idx2word)


class PoSDictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def __len__(self):
        return len(self.idx2word)


class NERDictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def __len__(self):
        return len(self.idx2word)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.limited_vocab = []

    def clean_word(self, word):
        return clean_str(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def limited_word(self, word):
        if word not in self.limited_vocab:
            self.limited_vocab.append(word)

    def remove_word(self, word):
        if word in self.word2idx:
            i = self.idx2word.index(word)
            del self.idx2word[i]
            self.idx2word.remove(word)
            self.word2idx.pop(word, None)

    """idx limit ensures that only the indices from inside the vocab are checked"""
    def replace_word(self, word, normalize=True):
        """uses levenshtein by default, no argument to avoid time spent on conditionals"""
        """edlib.align faster than levenshteinDistance"""
        if normalize:
            scores = []
            for vocab_word in self.limited_vocab:
                align_score = align(vocab_word, word)
                edit_distance = align_score['editDistance']
                if edit_distance != 0:
                    edit_distance /= align_score['alphabetLength']
                scores.append(edit_distance)
            scores = np.array([scores])
        else:
            scores = np.array([align(vocab_word, word)['editDistance'] for vocab_word in self.limited_vocab])
        replacement_word = self.idx2word[np.argmin(scores)]
        self.word2idx[word] = self.word2idx[replacement_word]

    def __len__(self):
        # was idx2word but changed for limword2idx assignment
        # changed to set since when limit is set most are given <unk>
        return len(set(list(self.word2idx.values())))


# this should map inds to pretrain inds
class PretrainedDictionary(object):
    def __init__(self, emb_type='word', nn=False):
        self.nn = nn
        self.word2idx = {}
        self.idx2word = []
        self.wv = {}
        self.embeddings = get_embeddings(emb_type)
        self.vocab_perc = 0

        if self.nn:
            self.id2neighbor = {}
            # word id -> neighbor id
            self.wid2nid = {}
            # I also need a map between word id and neighbor id because when setting nn.Embedding we
            # have not included neighor embeddings. This prob involves changing get_neighbors to return

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            id = len(self.idx2word) - 1
            self.word2idx[word] = id
            if word in self.embeddings:
                self.wv[id] = self.embeddings.wv[word]
                # cannot add neighbor ids while building vocab unless I am planning to
                #add all neighbors that are not necessarily words within the corpus.
                #if self.nn:
                #    # returns matrix with neighbor vector and sample probabilities
                #    self.id2neighbor[id] = get_nearest_neighbor(word, self.embeddings)
            else:
                self.vocab_perc +=1
                self.wv[id] = np.zeros((300,))
        return self.word2idx[word]

    def vocab_embeds(self, vocab_terms = None):
        if vocab_terms == None:
            vocab_terms = list(self.word2idx.keys())
        self.wv, self.word2id, self.id2word = get_vocab_embeddings(vocab_terms, self.wv)

    def __len__(self):
        return len(self.idx2word)


def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='../data/image_caption/coco/captions_train2014.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)

