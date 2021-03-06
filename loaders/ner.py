import torch.utils.data
import torch
from models.samplers.nnrs import NNRS
from models.samplers.transition import TransitionMatrix
from loaders.dictionary import NERDictionary
from torchtext import data
from torchtext import datasets
from loaders.other import embedding_dict
from macros import DATA_PATH
from torchtext import data
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vectors, GloVe, CharNGram

import logging



class NERCorpus(TransitionMatrix, NNRS):
    def __init__(self, emb_type=None, pretrain=False,
                 ns=False, reward=False, devicer='cpu'):

        self.dictionary = NERDictionary()
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
        self.NER_TAG = data.Field(init_token="<bos>", eos_token="<eos>")

        # Download and the load default data.
        train_data, val_data, test_data = datasets.sequence_tagging.SequenceTaggingDataset.splits(root=self.path,
                                                                fields=(('word', self.WORD), ('nertag', self.NER_TAG)))

        self.WORD.build_vocab(train_data.word, min_freq=3)
        self.dictionary.word2idx = dict(self.WORD.vocab.stoi)
        self.dictionary.idx2word = list(self.dictionary.word2idx.keys())
        self.NER_TAG.build_vocab(train_data.udtag)
        self.ner_vocab = self.NER_TAG.vocab.stoi

        if self.devicer == 'cpu':
            self.train, self.valid, self.test = data.BucketIterator.splits(
                (train_data, val_data, test_data), batch_size=batch_size, device=None)
        else:
            self.train, self.valid, self.test = data.BucketIterator.splits(
                (train_data, val_data, test_data), batch_size=batch_size, device=self.devicer)

            # set super class init parameters to be used for sequence replacement
        super().__init__(train=train_data, ner_tag=self.NER_TAG, devicer=self.devicer)

    def __len__(self):
        return len(self.WORD.vocab.itos)



logger = logging.getLogger(__name__)


class Ingredients(SequenceTaggingDataset):
    # New York Times ingredients dataset
    # Download original at https://github.com/NYTimes/ingredient-phrase-tagger

    urls = ['https://raw.githubusercontent.com/kolloldas/torchnlp/master/data/nyt/nyt_ingredients_ner.zip']
    dirname = ''
    name = 'nyt_ingredients_ner'

    @classmethod
    def splits(cls, fields, root=".data", train="train.txt",
               validation="valid.txt",
               test="test.txt", **kwargs):
        """Downloads and loads the NYT ingredients NER data in CoNLL format
        """

        return super(Ingredients, cls).splits(
            fields=fields, root=root, train=train, validation=validation,
            test=test, **kwargs)


def nyt_ingredients_ner_dataset(batch_size, use_local=False, root='.data/nyt_ingredients_ner',
                                train_file='train.txt',
                                validation_file='valid.txt',
                                test_file='test.txt',
                                convert_digits=True):
    """
    nyt_ingredients_ner: New York Times Ingredient tagging dataset
    Extract NYT ingredients dataset using torchtext. Applies GloVe 6B.200d and Char N-gram
    pretrained vectors. Also sets up per word character Field
    Parameters:
        batch_size: Batch size to return from iterator
        use_local: If True use local provided files (default False)
        root: Dataset root directory
        train_file: Train filename
        validation_file: Validation filename
        test_file: Test filename
        convert_digits: If True will convert numbers to single 0's

    Returns:
        A dict containing:
            task: 'nyt_ingredients.ner'
            iters: (train iter, validation iter, test iter)
            vocabs: (Inputs word vocabulary, Inputs character vocabulary,
                    Tag vocabulary )
    """

    # Setup fields with batch dimension first
    inputs_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, lower=True,
                             preprocessing=data.Pipeline(
                                 lambda w: '0' if convert_digits and w.isdigit() else w))

    inputs_char_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>",
                                     batch_first=True)

    inputs_char = data.NestedField(inputs_char_nesting,
                                   init_token="<bos>", eos_token="<eos>")

    labels = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)

    fields = ([(('inputs_word', 'inputs_char'), (inputs_word, inputs_char)),
               ('labels', labels)])

    # Load the data
    if use_local:
        train, val, test = SequenceTaggingDataset.splits(
            path=root,
            train=train_file,
            validation=validation_file,
            test=test_file,
            fields=tuple(fields))
    else:
        train, val, test = Ingredients.splits(fields=tuple(fields))

    logger.info('---------- NYT INGREDIENTS NER ---------')
    logger.info('Train size: %d' % (len(train)))
    logger.info('Validation size: %d' % (len(val)))
    logger.info('Test size: %d' % (len(test)))

    # Build vocab
    inputs_char.build_vocab(train.inputs_char, val.inputs_char, test.inputs_char)
    inputs_word.build_vocab(train.inputs_word, val.inputs_word, test.inputs_word, max_size=50000,
                            vectors=[GloVe(name='6B', dim='200'), CharNGram()])

    labels.build_vocab(train.labels)
    logger.info('Input vocab size:%d' % (len(inputs_word.vocab)))
    logger.info('Tagset size: %d' % (len(labels.vocab)))

    # Get iterators
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=batch_size,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    train_iter.repeat = False

    return {
        'task': 'nyt_ingredients.ner',
        'iters': (train_iter, val_iter, test_iter),
        'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab)
    }


if __name__ == "__main__":
    # remember this can work as a mtl problem where learn to predict the udtag and postag
    pc = NERCorpus()
    pc.get_data()
    for i, (batch) in enumerate(iter(pc.train)):
        print(batch.word)
        print(batch.udtag)
        break

