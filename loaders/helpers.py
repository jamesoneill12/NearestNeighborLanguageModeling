from macros import *
import pickle
from gensim.models import KeyedVectors
import numpy as np
import os
import torch
from collections import OrderedDict
from skipthoughts import UniSkip, BiSkip
from torch.autograd import Variable


def get_path(path_list, ext=''):
    path = "_".join([arg if type(arg) is str else str(arg)for arg in path_list])
    path = path.replace('./data/', '')
    if 'ptb' in path:
        path = PTB_WRITE_ROOT + ext + path + '.pickle'
    elif 'wikitext-2' in path:
        path = WIKI2_WRITE_ROOT + ext + path + '.pickle'
    elif 'wikitext-103' in path:
        path = WIKI3_WRITE_ROOT + ext + path + '.pickle'
    return path


def load_embeddings():
    google_model = KeyedVectors.load_word2vec_format(GOOGLE_VECTOR_PATH, binary=True)
    return google_model


def load_subword_embeddings(normalize=True, lower=True, clean_words=True):
    fasttext_model = KeyedVectors.load_word2vec_format(FASTTEXT_VECTOR_PATH)
    return fasttext_model


def get_embeddings(emb_type):
    return load_subword_embeddings() if 'sub' in emb_type else load_embeddings()


def get_vocab_embeddings(vocab_terms, pre_embs):
    print("Original vocabulary length : {}".format(len(vocab_terms)))
    word2vec = OrderedDict(
        {term: pre_embs.wv[term] for term in set(vocab_terms) if term in pre_embs.vocab})
    percentage_retrieved = 100 * len(word2vec) / float(len(vocab_terms))
    print("{} term vectors are available without spellchecking".format(percentage_retrieved))
    word2id = {word:i for i, word in enumerate(word2vec.keys())}
    id2word = {i:word for i, word in enumerate(word2vec.items())}
    return word2vec, word2id, id2word


def retrieve_embeddings(vocab_terms, emb_type = 'word'):

    pre_embs = get_embeddings(emb_type)
    term_vecs = get_vocab_embeddings(vocab_terms, pre_embs)
    fn = get_fn(emb_type)
    # need to load embedding not model !
    tuned_terms_vecs = load_model(fn)
    tuned_terms_vecs = OrderedDict(
        {term: tuned_terms_vecs.wv[term] for term in
         set(vocab_terms) if term in tuned_terms_vecs.vocab})
    term_vecs = [term_vecs, tuned_terms_vecs]

    return term_vecs, vocab_terms


def load_infersent_model(V=2, cuda=False):
    from models.networks.embeddings.sentence.infersent import InferSent
    MODEL_PATH="C:/Users/jamesoneill/Projects/embeddings/sentence/infersent"+str(V)+".pkl"
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    if cuda: infersent.cuda()
    return infersent


def load_skipthought(vocab, uni=True):
    dir_st = 'C:/Users/jamesoneill/Projects/embeddings/sentence/skipthought/'
    if uni:
        return UniSkip(dir_st, vocab)
    else:
        return BiSkip(dir_st, vocab)


def load_model(args, model, optimizer):
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    return model, optimizer


def get_fn(vec_name, wiki = 2):
    if vec_name == 'fasttext':
        if wiki == 2:
            return WIKI2_FASTTEXT_TRAINED_VECTOR_PATH
        else:
            return WIKI3_FASTTEXT_TRAINED_VECTOR_PATH
    elif vec_name == 'word2vec':
        if wiki == 2:
            return WIKI3_WORD2VEC_TRAINED_VECTOR_PATH
        else:
            return WIKI3_WORD2VEC_TRAINED_VECTOR_PATH


# passes a dictionary of attributes created from wiki2()
def save_wiki102_vocab(wiki2_vocab):
    paths = [WIKI2_WORD2IND_PATH, WIKI2_IND2WORD_PATH,
             WIKI2_WORD2VEC_VOCAB_PATH, WIKI2_ID2VEC_VOCAB_PATH]
    for path, (name, attrib) in zip(paths, wiki2_vocab.items()):
        save_vocab(attrib, path)


def get_vocab_embeddings(word2ind):
    embs = load_subword_embeddings()
    word2vec = {term: embs.wv[term] if term in embs.vocab else np.zeros((300,))
                for term, num in word2ind.items()}
    ind2vec = {num: word2ind[term] for term, num in word2ind.items() if term in word2vec}
    return word2vec, ind2vec


def save_vocab(terms, path=WIKI2_WORD2IND_PATH, show_len=True):
    if show_len:
        print("{} word in vocab.".format(len(terms)))
    with open(path, mode='wb') as pickle_file:
        pickle.dump(terms, pickle_file)


def load_vocab(path = WIKI2_WORD2IND_PATH):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    return data


def iterate_paths(paths):
    wiki_vocab = {}
    for path in paths:
        key = path.rsplit('/', 1)[1].replace('.p','')
        wiki_vocab[key] = load_vocab(path)


def load_wiki103_vocab():
    paths = [WIKI3_NEIGHBOR2VEC_PATH, WIKI3_WORD2IND_PATH,
             WIKI3_IND2WORD_PATH, WIKI3_WORD2VEC_VOCAB_PATH,
             WIKI3_ID2VEC_VOCAB_PATH]
    wiki_vocab = iterate_paths(paths)
    return wiki_vocab


def load_wiki102_vocab():
    paths = [WIKI2_NEIGHBOR2VEC_PATH, WIKI2_WORD2IND_PATH,
             WIKI2_IND2WORD_PATH, WIKI2_WORD2VEC_VOCAB_PATH,
             WIKI2_ID2VEC_VOCAB_PATH]
    wiki_vocab = iterate_paths(paths)
    return wiki_vocab


def old_saveandload():
    word2ind = load_vocab()
    id2word = load_vocab(WIKI2_IND2WORD_PATH)
    word2vec, id2vec = get_vocab_embeddings(word2ind)
    save_vocab(word2vec, WIKI2_WORD2VEC_VOCAB_PATH)
    save_vocab(id2vec, WIKI2_ID2VEC_VOCAB_PATH)


def padded_tensor(x, bsz):
    pad_tensor = torch.zeros((x.size(0), bsz - x.size(1))).type(torch.cuda.LongTensor)
    x = Variable(torch.cat([x, pad_tensor], 1))
    return x



