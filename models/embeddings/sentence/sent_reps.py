from .infersent import *
from skipthoughts import *
from loaders.helpers import load_skipthought, load_infersent_model
from macros import LARGE_FASTTEXT_VECTOR_PATH
from loaders.helpers import load_embeddings


def get_sent_reps(vocab, vocab_embs, reward):
    if reward == 'infersent':
        # assuming infersent v2 here
        sent_mod = load_infersent_model(cuda=True)
        sent_mod.set_w2v_path(LARGE_FASTTEXT_VECTOR_PATH)
        """ 
         figure out how to place vocab.idx2word into infer sent
         so that we don't have to pass all the sentences in again.
         Would be great if we could pass idxs2vecs instead to infersent
        """
        sent_mod_vocab = {vocab.idx2word[i]: vocab_embs[i, :] for i in range(vocab_embs.shape[0])}
        sent_mod.set_vocab(sent_mod_vocab)
        del sent_mod_vocab
        return sent_mod
    if reward == "uniskip":
        return load_skipthought(list(vocab.word2idx.keys()))
    elif reward == "biskip":
        return load_skipthought(list(vocab.word2idx.keys()), uni=False)
    elif reward == 'bleu':
        """implement raml stuff here"""
        pass
    elif reward == 'wmd':
        pre_emb = load_embeddings()
        pre_emb.init_sims(replace=True)
        vocab_embs = np.zeros((len(vocab), pre_emb.vector_size))
        for word, idx in vocab.word2idx.items():
            if word in pre_emb.vocab:
                vocab_embs[idx] = pre_emb[word]
            else:
                vocab_embs[idx] = np.zeros(pre_emb.vector_size)
        vocab.id2vec = torch.from_numpy(vocab_embs)
    else:
        print("sentence reps for {} not found or implemented".format(reward))
