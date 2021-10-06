import torch.utils.data
import torch
from models.samplers.nnrs import NNRS
from models.samplers.transition import TransitionMatrix
from loaders.dictionary import PoSDictionary
from torchtext import data
from torchtext import datasets
from loaders.other import embedding_dict
from macros import DATA_PATH


class PoSCorpus(TransitionMatrix, NNRS):
    def __init__(self, emb_type = None, pretrain=False,
                 ns=False, reward=False, devicer = 'cpu'):

        self.dictionary = PoSDictionary()
        self.devicer = torch.device("cuda" if devicer else "cpu")
        self.path = DATA_PATH

        # we need to do this for nsampling unless i'm resampling the input 
        # since we can replace the ud tags or ptb tags instead
        if ns or reward and pretrain is False:
            self.dictionary.wv = embedding_dict(self.dictionary.word2idx)
            print("{} words in vocab".format(len(self.dictionary.word2idx)))
            self.neighbourhood_dict = self.get_neighbor_dict()
        
    def get_data(self, batch_size=20):
        # Define the fields associated with the sequences.
        self.WORD = data.Field(init_token="<bos>", eos_token="<eos>")
        self.UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>")
        self.PTB_TAG = data.Field(init_token="<bos>", eos_token="<eos>")
        
        # Download and the load default data.
        train_data, val_data, test_data = datasets.UDPOS.splits(root=self.path,
            fields=(('word', self.WORD), ('udtag', self.UD_TAG), ('ptbtag', self.PTB_TAG)))

        self.WORD.build_vocab(train_data.word, min_freq=3)
        self.dictionary.word2idx = dict(self.WORD.vocab.stoi)
        self.dictionary.idx2word = list(self.dictionary.word2idx.keys())
        self.UD_TAG.build_vocab(train_data.udtag)
        self.ud_vocab = self.UD_TAG.vocab.stoi
        self.PTB_TAG.build_vocab(train_data.ptbtag)
        self.ptb_vocab = self.PTB_TAG.vocab.stoi

        if self.devicer == 'cpu':
            self.train, self.valid, self.test = data.BucketIterator.splits(
                (train_data, val_data, test_data), batch_size=batch_size, device = None)
        else:
            self.train, self.valid, self.test = data.BucketIterator.splits(
                (train_data, val_data, test_data), batch_size=batch_size, device=self.devicer)   

        # set super class init parameters to be used for sequence replacement
        super().__init__(train=train_data, ud_tag=self.UD_TAG, ptb_tag=self.PTB_TAG, devicer=self.devicer)

    def __len__(self):
        return len(self.WORD.vocab.itos)


if __name__ == "__main__":
    # remember this can work as a mtl problem where learn to predict the udtag and postag
    pc = PoSCorpus()
    pc.get_data()
    for i, (batch) in enumerate(iter(pc.train)):
        print(batch.word)
        print(batch.udtag)
        break

