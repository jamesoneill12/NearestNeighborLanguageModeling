PROJECTS_PATH = "C:/Users/jamesoneill/Projects/"
ROOT_PATH = PROJECTS_PATH + "NLP/GOLM/golm/golm/golm_hil/"
DATA_PATH = ROOT_PATH + "data/"
EMBEDDINGS_PATH = PROJECTS_PATH + "embeddings/"
MODEL_PATH = ROOT_PATH + "save_models/"
FASTTEXT_VECTOR_PATH = EMBEDDINGS_PATH + "wiki-news-300d-1M.vec"
LARGE_FASTTEXT_VECTOR_PATH = EMBEDDINGS_PATH + "crawl-300d-2M.vec"
GOOGLE_VECTOR_PATH = EMBEDDINGS_PATH + "GoogleNews-vectors-negative300.bin"

# ------------------- classifiers ----------------------------
LM_MODEL_PATH = MODEL_PATH + "lm/"
MT_MODEL_PATH = MODEL_PATH + "mt/"
POS_MODEL_PATH = MODEL_PATH + "pos/"

# ------------------- wiki-2 ----------------------------

WIKITEXT2_ROOT = 'C:/Users/jamesoneill/Projects/NLP/GOLM/golm/golm/golm_hil/data/wikitext-2/'
WIKITEXT2_TEST = WIKITEXT2_ROOT + "test.txt"
WIKITEXT2_TRAIN = WIKITEXT2_ROOT + "train.txt"
WIKITEXT2_VALID = WIKITEXT2_ROOT + "valid.txt"

WIKI2_NEIGHBOR2VEC_PATH = ROOT_PATH + "wiki2/neighbor2vec.p"
WIKI2_WORD2IND_PATH = ROOT_PATH + "wiki2/word2ind.p"
WIKI2_IND2WORD_PATH = ROOT_PATH + "wiki2/ind2word.p"
WIKI2_WORD2VEC_VOCAB_PATH = ROOT_PATH + "wiki2/word2vec.p"
WIKI2_ID2VEC_VOCAB_PATH = ROOT_PATH + "wiki2/id2vec.p"

WIKI2_CNN_MODEL_PATH = MODEL_PATH + "wiki2/cnn_model.p"
WIKI2_LSTM_MODEL_PATH = MODEL_PATH + "wiki2/lstm_model.p"
WIKI2_RNN_MODEL_PATH = MODEL_PATH + "wiki2/rnn_model.p"
WIKI2_GRU_MODEL_PATH = MODEL_PATH + "wiki2/gru_model.p"

WIKI2_FASTTEXT_TRAINED_VECTOR_PATH = ROOT_PATH + "wiki2/fasttext.vec"
WIKI2_WORD2VEC_TRAINED_VECTOR_PATH = ROOT_PATH + "wiki2/word2vec.bin"

WIKI2_KD_TREE_PATH = ROOT_PATH + "wiki2/kdt.p"
WIKI2_HDBSCAN_PATH = ROOT_PATH + "wiki2/hdbscan.p"


# ------------------- wiki-3 ----------------------------


WIKITEXT3_ROOT = 'C:/Users/jamesoneill/Projects/NLP/GOLM/golm/golm/golm_hil/data/wikitext-3/'
WIKITEXT3_TEST = WIKITEXT3_ROOT + "wiki.test.tokens"
WIKITEXT3_TRAIN = WIKITEXT3_ROOT + "wiki.train.tokens"
WIKITEXT3_VALID = WIKITEXT3_ROOT + "wiki.valid.tokens"

WIKI3_NEIGHBOR2VEC_PATH = ROOT_PATH + "wiki2/neighbor2vec.p"
WIKI3_WORD2IND_PATH = ROOT_PATH + "wiki3/word2ind.p"
WIKI3_IND2WORD_PATH = ROOT_PATH + "wiki3/ind2word.p"
WIKI3_WORD2VEC_VOCAB_PATH = ROOT_PATH + "wiki3/word2vec.p"
WIKI3_ID2VEC_VOCAB_PATH = ROOT_PATH + "wiki3/id2vec.p"

WIKI3_CNN_MODEL_PATH = MODEL_PATH + "wiki3/cnn_model.p"
WIKI3_LSTM_MODEL_PATH = MODEL_PATH + "wiki3/lstm_model.p"
WIKI3_RNN_MODEL_PATH = MODEL_PATH + "wiki3/rnn_model.p"
WIKI3_GRU_MODEL_PATH = MODEL_PATH + "wiki3/gru_model.p"

WIKI3_FASTTEXT_TRAINED_VECTOR_PATH = ROOT_PATH + "wiki3/fasttext.vec"
WIKI3_WORD2VEC_TRAINED_VECTOR_PATH = ROOT_PATH + "wiki3/word2vec.bin"

WIKI3_KD_TREE_PATH = ROOT_PATH + "wiki3/kdt.p"
WIKI3_HDBSCAN_PATH = ROOT_PATH + "wiki3/hdbscan.p"



# ------------------- PTB ----------------------------

PTB_ROOT = './data/ptb'
PTB_TEST = PTB_ROOT + "test.txt"
PTB_TRAIN = PTB_ROOT + "train.txt"
PTB_VALID = PTB_ROOT + "valid.txt"
PTB_NEIGHBOR2VEC_PATH = ROOT_PATH + "ptb/neighbor2vec.p"
PTB_WORD2IND_PATH = ROOT_PATH + "ptb/word2ind.p"
PTB_IND2WORD_PATH = ROOT_PATH + "ptb/ind2word.p"
PTB_WORD2VEC_VOCAB_PATH = ROOT_PATH + "ptb/word2vec.p"
PTB_ID2VEC_VOCAB_PATH = ROOT_PATH + "ptb/id2vec.p"
PTB_CNN_MODEL_PATH = MODEL_PATH + "ptb/cnn_model.p"
PTB_LSTM_MODEL_PATH = MODEL_PATH + "ptb/lstm_model.p"
PTB_RNN_MODEL_PATH = MODEL_PATH + "ptb/rnn_model.p"
PTB_GRU_MODEL_PATH = MODEL_PATH + "ptb/gru_model.p"
PTB_FASTTEXT_TRAINED_VECTOR_PATH = ROOT_PATH + "ptb/fasttext.vec"
PTB_WORD2VEC_TRAINED_VECTOR_PATH = ROOT_PATH + "ptb/word2vec.bin"
PTB_KD_TREE_PATH = ROOT_PATH + "ptb/kdt.p"
PTB_HDBSCAN_PATH = ROOT_PATH + "ptb/hdbscan.p"


# ------------------- POS ----------------------------

POS_ROOT = './data/udpos/en-ud-v2/en-ud-tag.v2.'
POS_TEST = POS_ROOT + "test.txt"
POS_TRAIN = POS_ROOT + "train.txt"
POS_VALID = POS_ROOT + "valid.txt"
POS_NEIGHBOR2VEC_PATH = ROOT_PATH + "neighbor2vec.p"
POS_WORD2IND_PATH = ROOT_PATH + "word2ind.p"
POS_IND2WORD_PATH = ROOT_PATH + "ind2word.p"
POS_WORD2VEC_VOCAB_PATH = ROOT_PATH + "word2vec.p"
POS_ID2VEC_VOCAB_PATH = ROOT_PATH + "id2vec.p"
POS_CNN_MODEL_PATH = MODEL_PATH + "cnn_model.p"
POS_LSTM_MODEL_PATH = MODEL_PATH + "lstm_model.p"
POS_RNN_MODEL_PATH = MODEL_PATH + "rnn_model.p"
POS_GRU_MODEL_PATH = MODEL_PATH + "pos/gru_model.p"
POS_FASTTEXT_TRAINED_VECTOR_PATH = ROOT_PATH + "fasttext.vec"
POS_WORD2VEC_TRAINED_VECTOR_PATH = ROOT_PATH + "word2vec.bin"
POS_KD_TREE_PATH = ROOT_PATH + "kdt.p"
POS_HDBSCAN_PATH = ROOT_PATH + "hdbscan.p"


# ------------------- MULTI ----------------------------

# ------------------- ISWLT ----------------------------

# ------------------- WMT ------------------------------


# ------------------- CHUNKING -------------------------

# ------------------- chunking ----------------------------

CHUNKING_ROOT = './data/chunking'
CHUNKING_TEST = CHUNKING_ROOT + "test.txt"
CHUNKING_TRAIN = CHUNKING_ROOT + "train.txt"

CHUNKING_NEIGHBOR2VEC_PATH = ROOT_PATH + "chunking/neighbor2vec.p"
CHUNKING_WORD2IND_PATH = ROOT_PATH + "chunking/word2ind.p"
CHUNKING_IND2WORD_PATH = ROOT_PATH + "chunking/ind2word.p"
CHUNKING_WORD2VEC_VOCAB_PATH = ROOT_PATH + "chunking/word2vec.p"
CHUNKING_ID2VEC_VOCAB_PATH = ROOT_PATH + "chunking/id2vec.p"

CHUNKING_CNN_MODEL_PATH = MODEL_PATH + "chunking/cnn_model.p"
CHUNKING_LSTM_MODEL_PATH = MODEL_PATH + "chunking/lstm_model.p"
CHUNKING_RNN_MODEL_PATH = MODEL_PATH + "chunking/rnn_model.p"
CHUNKING_GRU_MODEL_PATH = MODEL_PATH + "chunking/gru_model.p"

CHUNKING_FASTTEXT_TRAINED_VECTOR_PATH = ROOT_PATH + "chunking/fasttext.vec"
CHUNKING_WORD2VEC_TRAINED_VECTOR_PATH = ROOT_PATH + "chunking/word2vec.bin"

CHUNKING_KD_TREE_PATH = ROOT_PATH + "chunking/kdt.p"
CHUNKING_HDBSCAN_PATH = ROOT_PATH + "chunking/hdbscan.p"


# ------------------- Results ----------------------------


WRITE_ROOT = ROOT_PATH + "results/"
WIKI2_WRITE_ROOT = WRITE_ROOT + "wiki2/"
WIKI3_WRITE_ROOT = WRITE_ROOT + "wiki3/"
PTB_WRITE_ROOT = WRITE_ROOT + "ptb/"
CHUNKING_WRITE_ROOT = WRITE_ROOT + "chunking/"
WMT_WRITE_ROOT = WRITE_ROOT + "wmt/"
UDPOS_WRITE_ROOT = WRITE_ROOT + "udpos/"
COCO_WRITE_ROOT = WRITE_ROOT + "coco/"
MULTI_WRITE_ROOT = WRITE_ROOT + "multi/"
IWSLT_WRITE_ROOT = WRITE_ROOT + "iwlst/"
UDPOS_WRITE_ROOT = WRITE_ROOT + "udpos/"
SUMMARIZATION_WRITE_ROOT = WRITE_ROOT + "summarization/"