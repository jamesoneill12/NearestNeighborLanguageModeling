from models.samplers.transition import TransitionMatrix
from models.samplers.nnrs import NNRS
from loaders.dictionary import MTDictionary
from torchtext import data

from torchtext import datasets
from loaders.other import embedding_dict
from macros import DATA_PATH

import re
import en_core_web_sm
import de_core_news_sm

spacy_de = en_core_web_sm.load()
spacy_en = de_core_news_sm.load()

url = re.compile('(<url>.*</url>)')


def get_mt_data(arg):
    if arg.data in ['wmt', 'all']:
        print("Hello !!")
        corpus = WMTCorpus(batch_size=arg.batch_size, ns=arg.neighbor_sampler)
    elif arg.data in ['iwslt', 'all']:
        corpus = IWSLTCorpus(batch_size=arg.batch_size, ns=arg.neighbor_sampler)
    elif arg.data in ['multi', 'all']:
        corpus = MultiCorpus(batch_size=arg.batch_size, ns=arg.neighbor_sampler)
    else:
        raise ValueError("Could not find {}. Please choose wmt, iwslt, multi or all.".format(arg.data))
    return corpus


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]


class MTCorpus(TransitionMatrix, NNRS):
    def __init__(self, pretrain=False, ns=False, devicer='cuda'):

        # should there be seperate dicts for source and target ?
        self.dictionary = MTDictionary()
        self.devicer = devicer

        # we need to do this for nsampling unless i'm resampling the input
        # since we can replace the ud tags or ptb tags instead
        if ns and pretrain is False:
            self.dictionary.wv = embedding_dict(self.dictionary.word2idx)
            print("{} words in vocab".format(len(self.dictionary.word2idx)))
            self.neighbourhood_dict = self.get_neighbor_dict()

        self.BOS_WORD = u'<s>'
        self.EOS_WORD = u'</s>'
        self.BLANK_WORD = u"<blank>"

        self.DE = data.Field(tokenize=tokenize_de,  pad_token=self.BLANK_WORD, init_token=self.BOS_WORD,
                     eos_token=self.EOS_WORD)
        self.EN = data.Field(tokenize=tokenize_en, init_token=self.BOS_WORD,
                     eos_token=self.EOS_WORD, pad_token=self.BLANK_WORD)


        self.de_vocab = None
        self.en_vocab = None


class IWSLTCorpus(MTCorpus):

    def __init__(self, batch_size=20, devicer='cuda', pretrain=False, ns=False):
        super(IWSLTCorpus, self).__init__(ns=ns, pretrain=pretrain)

        train_data, val_data, test_data = datasets.IWSLT.splits(root=DATA_PATH+"iwslt/", exts=('.de', '.en'), fields=(self.DE, self.EN))
        self.train, self.valid, self.test = data.BucketIterator.splits(
            (train_data, val_data, test_data), batch_size=batch_size, device=devicer, repeat=False,
            shuffle=True, sort_within_batch=True, sort_key=lambda x: len(x.src))

        self.DE.build_vocab(train_data.src, min_freq=10, max_size=50000)
        self.EN.build_vocab(train_data.trg, min_freq=10, max_size=50000)

        # ud_tag in this case is the target vocab which is english in this case
        super(MTCorpus, self).__init__(ud_tag=self.EN, train=train_data)

        self.dictionary.en_word2idx = dict(self.EN.vocab.stoi)
        self.dictionary.de_word2idx = dict(self.DE.vocab.stoi)
        self.dictionary.en_idx2word = list(self.dictionary.en_word2idx.keys())
        self.dictionary.de_idx2word = list(self.dictionary.de_word2idx.keys())
        self.de_vocab = self.DE.vocab.stoi
        self.en_vocab = self.EN.vocab.stoi


class WMTCorpus(MTCorpus):

    def __init__(self, batch_size=20, devicer='cuda', pretrain=False, ns=False):
        super(WMTCorpus, self).__init__(ns=ns, pretrain=pretrain)

        # Download and the load default data.
        train = 'train.tok.clean.bpe.32000'
        validation = 'newstest2013.tok.bpe.32000'
        test = 'newstest2014.tok.bpe.32000'

        train_data, val_data, test_data = datasets.WMT14.splits(root=DATA_PATH+"wmt/",
                            train=train, validation=validation, test=test, exts=('.de', '.en'), fields=(self.DE, self.EN))

        self.train, self.valid, self.test = data.BucketIterator.splits(
            (train_data, val_data, test_data), batch_size=batch_size, device=devicer, repeat=False,
                shuffle=True, sort_within_batch=True, sort_key=lambda x: len(x.src))

        self.DE.build_vocab(train_data.src, min_freq=3)
        self.EN.build_vocab(train_data.trg, max_size=50000)

        self.dictionary.word2idx = dict(self.DE.vocab.stoi)
        self.dictionary.idx2word = list(self.dictionary.word2idx.keys())
        self.de_vocab = self.DE.vocab.stoi
        self.en_vocab = self.EN.vocab.stoi
        # ud_tag in this case is the target vocab which is english in this case
        super(MTCorpus, self).__init__(ud_tag=self.EN, train=train_data)

    def __len__(self):
        return len(self.EN.vocab.itos)


class MultiCorpus(MTCorpus):

    def __init__(self, batch_size=20, devicer='cuda', pretrain=False, ns=False):
        super(MultiCorpus, self).__init__(pretrain=pretrain, ns=ns)

        # Download and the load default data.
        train_data, val_data, test_data = datasets.Multi30k.splits(
            root=DATA_PATH+"multi/", exts=('.de', '.en'), fields=(self.DE, self.EN))

        self.train, self.valid, self.test = data.BucketIterator.splits(
            (train_data, val_data, test_data), batch_size=batch_size, device=devicer, repeat=False,
            shuffle=True, sort_within_batch=True, sort_key=lambda x: len(x.src))

        self.DE.build_vocab(train_data.src, min_freq=2)
        self.EN.build_vocab(train_data.trg, max_size=50000)

        self.dictionary.word2idx = dict(self.DE.vocab.stoi)
        self.dictionary.idx2word = list(self.dictionary.word2idx.keys())
        self.de_vocab = self.DE.vocab.stoi
        self.en_vocab = self.EN.vocab.stoi

        # ud_tag in this case is the target vocab which is english in this case
        super(MTCorpus, self).__init__(ud_tag=self.DE, train=train_data)

    def __len__(self):
        return len(self.EN.vocab.itos)


if __name__ == "__main__":
    # remember this can work as a mtl problem where learn to predict the udtag and postag
    pc = MultiCorpus()
    train_data = pc.train
    cnt = 0
    for i, batch in enumerate(train_data):
        print("source:{} \t  target {}".format(batch.src.size(), batch.trg.size()))
        cnt += batch.src.size(1)
    print("train data length")
    print(cnt)

