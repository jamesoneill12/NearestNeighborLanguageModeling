
import argparse
from macros import WIKITEXT2_ROOT, WIKITEXT3_ROOT, PTB_ROOT, POS_ROOT

MODEL = 'LSTM'

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 and Wikitext-3 RNN/LSTM Language Model')

# training, data and preprocess params
parser.add_argument('--data', type=str, default=WIKITEXT2_ROOT,
                    help='location of the data corpus')
parser.add_argument('--vocab_limit', type=int, default=int(5e4),
                    help='vocabulary limit (important for en8 and wiki103)')
parser.add_argument('--lcs', type=bool, default=False,
                    help='if true, it tries to assign idx for oov words which have the closes lcs'
                         'to words within the vocabulary limit based on term frequency')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,help='sequence length')


# task params
# this decides the multi-task learning aspect if pos is on we learn pos with nlm
parser.add_argument('--pos', default= False,
                    help='if we true, we let know we are training pos tagger')
parser.add_argument('--nptb_token', default=False,
                    help='if true, we train ptb and ud tags simulataneously')
parser.add_argument("--translation", default=None, type=str, help="choices: "
                    "multi: The small-dataset WMT 2016 multimodal task, also known as Flickr30k"
                    "wlst: The IWSLT 2016 TED talk translation task "
                    "wmt: The WMT 2014 English-German dataset, as preprocessed by Google Brain.")
parser.add_argument("--sim_mt", default=False, type=bool,
                    help="changes scheduled sampling to not using last few tokens if true")
parser.add_argument("--joint_mt", default=False, type=bool, help="trains a source language model also if true")
parser.add_argument("--qa", default=None, type=str, help="choices:"
                    "SQuAD: Stanford Question Answering Dataset"
                    "BaBI: Facebook Question Answering")

# posterior params
# if vocab hierarchy is on, this turns cosine loss on and if scheduled sampler is turned on also then
# the scheduled sampling is used with hierarchy method
parser.add_argument('--vocab_hierarchy', type =  bool, default = False,
                    help='if true, we predict embedding and lookup in the hierarchy')
"""
parser.add_argument('--hsoftmax', type =  bool, default = False,
                    help = 'if true, uses hierarchical softmax')
                    """
parser.add_argument('--approx_softmax', type = str, default = None,
                    help = 'bit of refactor needed here because some options are not softmax approximations'
                           'hsoftmax: predicts using hierarhical softmax,'
                           'soft_mix: uses a mixture of softmaxes '
                           'hsoft_mix: uses a mixture of softmaxes in hierarchial softmax'
                           'hsoft_mix_tuned: puts attention on each softmax mixture'
                           'relaxed_hsoftmax: gumbel softmax on hsoftmax'
                           'dsoftmax: predicts using differentiated softmax (not implemented),'
                           'dsoft_mix: mixture of differentiated softmaxes'
                           'nce: noise contrastive estimation model instead,'
                           'adasoftmax: uses adaptive softmax '
                           'relaxed_ecoc: use gumbel ecoc and sample latent codes'
                           'ecoc_mix: mixture of latent codes'
                           'ecoc_mix: tunable mixture of latent codes'
                           'target_sampling:'
                           'sampled_softmax: uses lei mao version of gumbel softmax'
                    )


parser.add_argument('--noise', default = None,
                    help = "used for nce")
parser.add_argument('--index-module', type=str, default='linear',
                    help='index module to use in NCELoss wrapper')
parser.add_argument('--noise-ratio', type=int, default=10,
                    help='set the noise ratio of NCE sampling, the noise'
                         ' is shared among batch')
parser.add_argument('--norm-term', type=int, default=9,
                    help='set the log normalization term of NCE sampling')


# if i only keep indices of vocab then won't be able to replace neighbor that is not also in vocab
# need to decide if I should include all neighbors in vocab also, but this will result in a massive
# vocab proportional to the number of neighbors included


# gradient and sampling params
parser.add_argument('--neighbor_sampler', type =  bool, default = False,
                    help='if true, we include neighbor embeddings as the target')
parser.add_argument('--neighbor_prob', type =  float, default = 0.0,
                    help='starts at 0 probability')
parser.add_argument('--ns_prob', type =  float, default=0.0,
                    help='if true and neighbor sampler true, with small prob we choose neighbor')
parser.add_argument('--ns_uthresh', type =  float, default = 0.5,
                    help='upper threshold for sample probability')
"""
parser.add_argument('--scheduled_sampler', default=None, type = str,
                    help = 'fixed (stays at ss_prob), linear_decay, exp_decay or sigmoid_decay')
"""
parser.add_argument('--diff_ss', default=False, type = bool,
                    help='If true, we use differentiable scheduled sampling,'
                         ' which can be found as function in the RNN class')

parser.add_argument('--scheduled_sampler', default=None, type = str,
                    help='rate of sample prob change: very fast, fast, nearly_linear, linear, sigmoid, very slow'
                        'probabily better to use sigmoid or very slow which only let ')
parser.add_argument('--ss_prob', default=0.0, type = float,
                    help='should probably always start off with 0.0')
parser.add_argument('--ss_uthresh', default=0.2, type = float,
                    help='upper probability threshold for scheduled sampling strategy')
parser.add_argument('--ss_emb', default=False, type = bool,
                    help='by default, ss uses past predicted decoder embedding instead'
                         ' of the input encoder embedding. ss_emb=True ensures encoder embedding used')
parser.add_argument('--ss_soft_emb', default=False, type = bool,
                    help='ss_soft_emb=True linear combination of top k predicted embeddings')
parser.add_argument('--cw_mix', default=False, type = bool,
                    help='use codeword mixture scheduled sampling whereby '
                         'predicted ecoc is mixed with the true target given ss curriculum')



parser.add_argument('--structured_output', default=None,
                    help='kd-tree or hdbscan defines hierarchy of the output')
# rl learning and reinforcement learning alternaties to sampling and supervised learning methods
parser.add_argument('--reinforce', default=False, type=bool,
                    help='if reinforce is used for stochastic gradient estimation we do not use ce, ss etc.')
parser.add_argument('--reparam_trick', default=False, type=bool,
                    help='if reparam_trick is used we use the pathwise derivative scoring function.')
parser.add_argument('--gumbell', default=False, type=bool,
                    help='if gumbell is used we sample from this')
parser.add_argument('--actor_critic', default=False, type=bool,
                    help='if actor_critic used, we use pg seq2seq net.')

# network params
parser.add_argument('--model', type=str, default=MODEL,
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
# if pretrain true make sure to keep emsize and nhid = 300
parser.add_argument('--pretrained', default=None,
                    help='size of word embeddings: None, word or subword')
parser.add_argument('--target_pretrained', default=None,
                    help='in the case of machine translation where there is two vocabularies'
                         'size of word embeddings: None, word or subword')
parser.add_argument('--tune', default= False,
                    help='tunes pretrained embeddings if set true')
parser.add_argument('--nhid', type=int, default=1100,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')

# regularization params
parser.add_argument('--batch_norm', type=bool, default=False,
                    help='batch_norm applied to input and output layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropout_method', type=str, default='standard',
                    help='standard, gaussian, variational, concrete, curriculum')
parser.add_argument('--dropout_lb', type=float, default=0.0,
                     help='lower bound for curriculum dropout')
parser.add_argument('--dropout_ub', type=float, default=0.8,
                     help='upper bound for curriculum dropout')
parser.add_argument('--dropout_position', type=int, default=1,
                    help='1=drop_embedding, 2=drop_output, 3=1and2')
parser.add_argument('--fixed_dropout', type=bool, default=False,
                    help='if true, it fixes the mask across timesteps for all variations (including concrete)')
parser.add_argument('--dropconnect', type=float, default=None,
                    help='dropping out weights instead of activations')
parser.add_argument('--dropconnect_lb', type=float, default=0.0,
                     help='lower bound for curriculum dropconnect')
parser.add_argument('--dropconnect_ub', type=float, default=0.8,
                     help='upper bound for curriculum dropconnect')
parser.add_argument('--alpha', type=float, default=0.2,
                     help='fraternal dropout: L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=0.2,
                     help='fraternal dropout: slowness regularization applied on RNN activiation (beta = 0 means no regularization)')

parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--dec_nhid', type=int, default=200,
                    help='tie the word embedding and softmax weights')
parser.add_argument('--dec_output_size', type=int, default=200,
                    help='decoder output size')
parser.add_argument('--attention', type=bool, default=False,
                    help='If true, turned on for NMT and can be used with SS and NNRS')

# latent neural language modelling
parser.add_argument('--latent', type=str, default=False,
                    help='if true, we train to predict pretrained embedding instead of tokens')
parser.add_argument('--latent_k', type=int, default=1,
                    help='num of prediction samples to draw when comparing to target vector/vectors')
parser.add_argument('--latent_nn', type=int, default=False,
                    help='if true uses nearest vocabulary vector as prediction')


# loss and optimization params
parser.add_argument('--loss', type=str, default='ce',
                    help='ce=log-likelihood, '
                         'bce = binary cross-entropy, in which case error-correcting codes are used to represent the output'
                         'neighbor_ce=log-likelihood for multi-label classification'
                         ' where the labels are the target neighbors'
                         'rce = reward augmented log-likelihood with ce for testing,'
                         'wmd_ce = word movers distance augmented maximum likelihood'
                         'rce_neighbor=same as neighbor_ce except it assigns '
                         'scaled exponentially decaying weights to neighbors based on cosine sim.'
                         'rce_alt=reward augmented log-likelihood with cosine for testing,'
                         'jsd=jenson-shannon, '
                         'mmd=maximum mean discrep, '
                         'cosine=squared cosine proximity'
                    )

parser.add_argument('--codebook', type=bool, default=False,
                    help='If true, it activates the codebook for NORMAL CE training,'
                         ' only to be used for calculating hamming distances between '
                         'predictions and targets and NOT ecoc approach')

parser.add_argument('--optimizer', type=str, default='amsgrad',
                    help='when set to None, it picks vanilla sgd with decayed learning rate')
parser.add_argument('--dec_optimizer', type=str, default='amsgrad',
                    help='when set to None, it picks vanilla sgd with decayed learning rate')
parser.add_argument('--scheduler', type=str, default='cosine_anneal',
                    help='None, cosine_anneal, lro, multi_step')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--decay_factor', type=float, default=4,
                    help='initial learning rate')
parser.add_argument('--lr_anneal', type=bool, default=True,
                    help='if true anneal the learning rate')
parser.add_argument('--dec_lr', type=float, default=0.01,
                    help='initial decoder learning rate')
parser.add_argument('--control', type=bool, default=True,
                    help='when set true and standard sgd with annealing is used, it allows the learning rate'\
                    'to increase if improvements are made over 2 epochs once decreased')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--adversarial', type=bool, default=False,
                    help='use adversarial training')
parser.add_argument('--adv_loss', type=str, default='bce',
                    help='bce: binary cross-entropy loss')
parser.add_argument('--gen_lr', type=float, default=5,
                    help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999,
                    help='adam: decay of first order momentum of gradient')


# predict params
parser.add_argument('--search', default=None,
                    help="beam and possible other options to consider such as monte carlo tree search")
parser.add_argument('--beam_width', default=10, type = int,
                    help='chooses beam width if chosen by search')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

# utils params
parser.add_argument('--cuda', default=True, action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save_model', type=bool, default=True,
                    help='whether to save the model or not')
parser.add_argument('--save_test_batch', type=bool, default=False,
                    help='save numpy array of target indices with vocab')
parser.add_argument('--save', type=str, default='./save_models/'+MODEL.lower()+'.pt',
                    help='path to save the final model')
parser.add_argument('--save_losses', type=bool, default=False,
                    help='stores probabilities for each token in a sequence'
                         'so we can compare variance as a function of time')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

# print params
parser.add_argument('--check_grad', default=False, type=bool,
                    help='prints out gradients of network to make sure gradients are being updated for all parameters')

# evaluation params
parser.add_argument('--accuracy', default=False, type=bool,
                    help='by default perplexity, but if true it stores train/val/test accuracies in performance dict')
parser.add_argument('--bleu', default=False, type=bool,
                    help='includes bleu in evaluation')
parser.add_argument('--wmd', default=False, type=bool,
                    help='includes wmd between embeddings')
parser.add_argument('--sinkhorn_iter', default=100, type=int,
                    help='when wmd is used as loss between pretrained embeddings')
parser.add_argument('--sinkhorn_eps', default=0.1, type=float,
                    help='when wmd is used as loss between pretrained embeddings')

args = parser.parse_args()
