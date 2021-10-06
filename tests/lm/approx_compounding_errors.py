"""
    carries out tests where we train the language model
    using error-correcting output codes

    Decision to be made:
        If scheduled sampling is chosen while using error codes then we have 2 options:
            (1) from the nbits convert back to index and use that index to look up the embedding matrix
                which is |V| dimensionality.
            (2) OR, pass the predicted binary code as input to embedding matrix which is also a binary coded
                lookup matrix.

    Not sure which of the two is the best approach. I would say (1) because the dimensionality of the input is very low
    for the embedding input in the case of (2).

    WHAT SETTINGS ACTIVATE ECOC ?
        data.Corpus(error_coding=True)
        if loss = bce

"""

import os
from settings import *
from utils.getters import get_ecoc_corpus
from trainers.lm.lm_train import run
from loaders.helpers import get_path

# shouldn't both looping ns uthreshs if neighbor_sampler is False,
# should probably just get rid of args.neighbor_sampler
datas = [PTB_ROOT, WIKITEXT2_ROOT] #, WIKITEXT3_ROOT]
models = ['LSTM'] # , 'GRU'] # 'HIGHWAY' QRNN, 'GORU'
drop_methods = ['standard']
drop_schedules = None

args.epochs = 40
args.ns_prob = 0.0
args.ns_uthresh = 0.0

"""MAIN DIFFERENCE: WE ACTIVATE OCOC BY SETTING LOSS TO BCE"""
args.control = False
args.vocab_hierarchy = False
args.neighbor_sampler = False  # [False, True]
args.save_model = False
# consider testing batch_norm = True, particularly for highway networks.
args.batch_norm = False
args.dropconnect = False
"""for the moment, makes sure concrete is fixed across timesteps"""
args.fixed_dropout = True
args.save_losses = True
"""In the context of this paper ss_uthresh is the upper bound on the dropout probability"""
args.ss_uthresh = None # was 0.8
args.pretrain = None
args.optimizer = None
args.scheduler = None

methods = ["ss-hs", "ss-as", "ss-hs-soft", "ss-as-soft",
            "ss-ecoc", "cms-ecoc", "ss-ecoc-soft", "cms-ecoc-soft"]
args.emsize = 400
args.nhid = 400

for text in datas:
    already_loaded = False
    args.data = text
    args.neighbor_sampler = False
    args, corpus = get_ecoc_corpus(text, args)
    for method in methods:
        """first tests ce with codebook so to evaluate with hamming distance and accuracy"""
        """then tests ecoc and also evaluates with hamming distance and accuracy """
        if "ecoc" in method:
            args.codebook = True
            ext = "ecoc/"
            losses = ["bce"]  # "ce",
            if method == "ss-ecoc":
                args.ss_emb = False
                args.cw_mix =  False # when one is on, the other should be off
                # by default cw_mix curriculum is exponential with 0.5 upper threshold
            elif method == "cms-ecoc":
                args.ss_emb = False
                args.cw_mix =  True
            elif method == "ss-ecoc-soft":
                args.ss_emb = True
                args.cw_mix =  False
            elif method == "cms-ecoc-soft":
                args.ss_emb = True
                args.cw_mix =  True
            if already_loaded is False:
                args, corpus = get_ecoc_corpus(text, args)
                already_loaded = True
        else:
            if method == "ss-hs":
                losses = ["bce"]  # "ce",
                args.codebook = True
                ext = "ecoc/"
                args.ss_emb = False
                args.cw_mix = False  # when one is on, the other should be off
            elif method == "ss-as":
                losses = ["bce"]  # "ce",
                args.codebook = True
                ext = "ecoc/"
                args.ss_emb = False
                args.cw_mix = False  # when one is on, the other should be off


        for drop_method in drop_methods:
            args.dropout_method = drop_method
            for model in models:
                args.model = model
                print("{}-{}-{}".format(text, model, method))
                path_attribs = [text, model, drop_method, args.dropout, args.loss]
                args.results_path = get_path(path_attribs, ext)
                if os.path.exists(args.results_path) is False:
                    run(args, corpus)

