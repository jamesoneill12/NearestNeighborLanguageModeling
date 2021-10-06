# coding: utf-8

import torch.onnx
from models.networks.recurrent.rnn import RNNModel
from loaders.data import build_unigram_noise
from utils.metrics.dist import hamming_distance
from utils.eval.accuracy import get_accuracy, CodeAccuracy
from trainers.lm.train_helpers import update_performance, add_codebook_eval_metrics
from utils.eval.bleu import get_bleu
from utils.eval.wmd import get_wmd
from loaders.helpers import save_vocab, load_embeddings
from models.samplers.ss import update_ss_prob
from models.loss.loss import get_criterion, KL
from utils.batchers import *
import random


# target corpus is now added in the case of seq2seq
def run(args, corpus, dec_corpus=None):
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Device: {}".format(device))

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.pretrained is not None:
        args.emsize = 300
        args.nhid = 300

    ###############################################################################
    # Load data
    ###############################################################################

    # eval batch size is actually the sequence length so keep at 10
    eval_batch_size = args.bptt # 10 if args.approx_softmax != "adasoftmax" else args.batch_size

    if 'pos' in args.data:
        train_data = corpus.train
        val_data = corpus.valid
        test_data = corpus.test
        ntokens = len(corpus.dictionary)
        nud_tags = len(corpus.ud_vocab)
        nptb_tags = len(corpus.ptb_vocab)
        nptb_tags = nptb_tags if args.nptb_token else None
        print("Number of ptb pos tags: {0} \t ud tags {1}:".format(nptb_tags, nud_tags))
    else:
        train_data = batchify(corpus.train, args.batch_size, device)
        val_data = batchify(corpus.valid, eval_batch_size, device)
        test_data = batchify(corpus.test, eval_batch_size, device)
        ntokens = len(corpus.dictionary)
        nud_tags = None
        nptb_tags = None
        print("Number of Tokens: {}".format(ntokens))

    ###############################################################################
    # Build the model
    ###############################################################################

    embeddings = corpus.dictionary.wv if args.pretrained is not None else None
    # ecoc_nbits None if loss!="bce"
    if hasattr(corpus, 'error_codes') and args.loss == "bce":
        ecoc_nbits = corpus.error_codes.size(1)
        cacc = CodeAccuracy(ecoc_nbits=ecoc_nbits)
    else:
        ecoc_nbits = None

    if args.approx_softmax == "nce":
        freq = torch.FloatTensor(list(corpus.token_count.values()))
        freq = build_unigram_noise(freq) #, not needed, done in AliasSampling
    else:
        freq = None

    # if use cw sampling we need the codebook, if we standard ss with ecoc we need codebook
    if args.cw_mix or (args.loss == "bce" and args.scheduled_sampler is not None):
        # most sampling done near end of training
        args.scheduled_sampler = 'very slow'
        # args.codebook = corpus.error_codes
        args.ss_uthresh = 0.6

    # codebook=args.codebook,
    model = RNNModel(rnn_type=args.model, ntoken=ntokens, ninp=args.emsize, nhid=args.nhid, approx_softmax=args.approx_softmax,
                     noise_ratio=args.noise_ratio, norm_term=args.norm_term, nlayers=args.nlayers, ss_emb=args.ss_emb,
                     cw_mix=args.cw_mix, bsize=args.batch_size, pretrained=embeddings, drop_rate=args.dropout, unigram_dist=freq,
                     drop_method=args.dropout_method, drop_position=args.dropout_position,  ecoc_nbits=ecoc_nbits,
                     fixed_drop=args.fixed_dropout, dropc=args.dropconnect, pos=args.pos, nud_tags=nud_tags, nptb_tags=nptb_tags,
                     batch_norm=args.batch_norm, tie_weights=args.tied, alpha=args.alpha, beta=args.beta).to(device)


    if args.approx_softmax == "adasoftmax":
        # 1000, 3000, 10000
        cutoff = [int(ntokens/10), int(ntokens/3), ntokens]
        args.decay_factor = 8
    else:
        cutoff = None

    if args.approx_softmax == "adasoftmax":
        print("Cutoff: - {}".format(cutoff))

    if "relaxed_ecoc" == args.approx_softmax or\
            "relaxed_softmax"  == args.approx_softmax or \
            "relaxed_hsoftmax"  == args.approx_softmax:
        # ensures cross entropy for gumbel logit/softmax
        criterion = get_criterion(args.loss, rs=True)
    elif args.approx_softmax != "nce":
        """criterion inside RNNModel when nce is used"""
        if args.approx_softmax == "adasoftmax": args.loss = "ada"
        criterion = get_criterion(args.loss, cutoff=cutoff, noise=freq)

    if 'rce' in args.loss:
        class_weights = [1] * ntokens
        vocab_vectors = corpus.dictionary.wv
        criterion = criterion(class_weights=class_weights, batch_size=args.batch_size,
                              lm=True, vocab_vectors=vocab_vectors, temp=10)
        eval_criterion = criterion if 'alt' in args.loss else nn.CrossEntropyLoss()

    if args.optimizer == 'amsgrad':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd' or args.optimizer is None:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr)

    ###############################################################################
    # Training code
    ###############################################################################

    measure = 'ppl' if 'pos' not in args.data else 'acc'
    performance = {'train_epoch': [], 'train_loss': [], 'train_' + measure: [], 'train_lr': [],
                   'val_epoch': [], 'val_loss': [], 'val_' + measure: [], 'val_lr': [],
                   'test_loss': [], 'test_' + measure: []}

    if args.accuracy: performance['train_acc'] = [];performance['val_acc'] = []; performance['test_acc'] = []
    if args.bleu:
        performance['train_bleu_quality'] = [];performance['val_bleu_quality'] = []; performance['test_bleu_quality'] = []
        performance['train_bleu_diversity'] = [];performance['val_bleu_diversity'] = []; performance['test_bleu_diversity'] = []

    if args.wmd:
        performance['train_wmd_quality'] = [];performance['val_wmd_quality'] = []; performance['test_wmd_quality'] = []
        performance['train_wmd_diversity'] = [];performance['val_wmd_diversity'] = []; performance['test_wmd_diversity'] = []
        pre_emb = load_embeddings()
        pre_emb.init_sims(replace=True)
        vocab_embs = np.zeros((len(corpus.dictionary), pre_emb.vector_size))
        for word, idx in corpus.dictionary.word2idx.items():
            if word in pre_emb.vocab:
                vocab_embs[idx] = pre_emb[word]
            else:
                vocab_embs[idx] = np.zeros(pre_emb.vector_size)
        corpus.dictionary.id2vec = torch.from_numpy(vocab_embs)
        # del pre_emb

    """ If codebook true, it saves the results for standard training """
    if args.codebook or args.loss == 'bce':
        performance = add_codebook_eval_metrics(performance)
        t = Variable(torch.Tensor([0.5])).cuda()  # threshold

    if args.fixed_dropout and args.dropout_method == 'concrete':
        if args.dropout_position == 1 or args.dropout_position == 3:
            performance['drop_in_rate'] = []
        if args.dropout_position == 2 or args.dropout_position == 3:
            performance['drop_out_rate'] = []

    """ uncertainty saves the calibrated output probabilities """
    def evaluate(data_source, save_losses=False, data_split="val"):

        # Turn on evaluation mode which disables dropout.
        model.eval()

        total_loss = 0.
        if args.codebook or args.loss == "bce":
            total_acc = 0.
            total_ham = 0.
        elif args.accuracy: total_acc = 0.
        if args.wmd: total_q_wmd, total_d_wmd = 0., 0.
        if args.bleu: total_q_bleu, total_d_bleu = 0., 0.

        if args.save_test_batch:
            test_batch = []
            test_batcher = {}

        num_tokens = len(corpus.dictionary)
        hidden = model.init_hidden(eval_batch_size)

        if save_losses:
            loss_per_tstep = []
            """This basically gets cross entropy for each time steps
             for each sequence so we can analyze each step"""
            uncertainty_criterion = get_criterion('ce_all')
            weights = torch.ones(num_tokens)
            un_crit = uncertainty_criterion(weights)

        with torch.no_grad():

            for i in range(0, data_source.size(0) - 1, args.bptt):

                data, targets = get_batch(data_source, i, args.bptt)

                if args.loss == 'bce':
                    targ_ind = targets.clone()
                    target_codes = corpus.error_codes[targets.flatten()]

                if args.approx_softmax == "hsoftmax" or args.approx_softmax == "adasoftmax" \
                        or (args.loss == "bce" and args.cw_mix):
                    """works for hsoftmax and adasoftmax"""
                    output, hidden = model(data, hidden, targets)
                else:
                    output, hidden = model(data, hidden)
                    # if args.loss == 'bce':
                        # removed ntokens here because should be 14 instead of 10000 for PTB etc.
                        # output_flat = output.view(output.size(0) * output.size(1), -1)
                    #else:
                    output_flat = output.view(output.size(0) * output.size(1), -1)

                if args.loss != 'bce':
                    if args.codebook:
                        yhat = output.view(output.size(0) * output.size(1), -1)
                        """ Here I need to compute perplexities while taking
                            the error-checks into account
                        """
                        ppl = corpus.codebook.get_perplexity(yhat, targets)

                        loss = criterion(yhat, targets)
                        yhat_val, yhat_ind = yhat.max(1)
                        predicted_codes, target_codes = corpus.codebook. \
                            lookup_codes(corpus.error_codes, yhat_ind, targets)
                        ham_dist = hamming_distance(predicted_codes, target_codes)
                        accuracy = get_accuracy(yhat_ind, targets)
                        total_acc += accuracy
                        total_ham += ham_dist
                    else:
                        if args.approx_softmax == "hsoftmax":
                            loss = -torch.mean(torch.log(output))
                        elif args.approx_softmax == "adasoftmax":
                            loss = criterion(output, targets)
                        elif args.approx_softmax == "nce":
                            loss = model.nce_loss(output, targets.view(output.size(0), output.size(1)))
                        else:
                            if args.loss == 'rce':
                                loss = eval_criterion(output_flat, targets) * criterion(output_flat, targets)
                                loss += criterion(yhat, targets)
                            else:
                                yhat = output.view(-1, ntokens)
                                loss = criterion(yhat, targets)

                        if args.accuracy:
                            # should fix for  "adasoftmax"
                            if args.approx_softmax in ["hsoftmax", "nce"]:
                                if args.approx_softmax == "nce":
                                    targets = targets.view(output.size())
                                accuracy = get_accuracy(output, targets)
                            elif args.loss == "rce":
                                accuracy = get_accuracy(output_flat, targets)
                            else:
                                accuracy = get_accuracy(yhat, targets)
                            total_acc += accuracy

                        if args.bleu or args.wmd:
                            output_val, output_ind = output.max(1)
                            out_b = output_ind.view(data.size()).t().cpu().numpy()
                            targ_b = targets.view(data.size()).t().cpu().numpy()
                            out_b_new = out_b.copy()
                            np.random.shuffle(out_b_new)
                            if args.bleu:
                                total_q_bleu += get_bleu(out_b, targ_b)
                                total_d_bleu += get_bleu(out_b, out_b_new)
                            if args.wmd:
                                total_d_wmd += get_wmd(out_b, targ_b, corpus, pre_emb)
                                total_q_wmd += get_wmd(out_b, out_b_new, corpus, pre_emb)

                else:
                    yhat = output.view(output.size(0) * output.size(1), -1)
                    loss = criterion(yhat, target_codes)
                    # if args.codebook:
                    # no need for a conditional, if args.loss == bce then
                    # codebook is assumed
                    # target_codes different to above targets because they are rounded
                    _, target_codes = corpus.codebook. \
                        lookup_codes(corpus.error_codes, yhat, targets)
                    predicted_codes = (yhat > t).float() * 1

                    ham_dist = hamming_distance(predicted_codes, target_codes)
                    accuracy = cacc.get_code_accuracy(predicted_codes, targ_ind)
                    total_acc += accuracy
                    total_ham += ham_dist

                if save_losses and ecoc_nbits is None and args.approx_softmax == "adasoftmax":
                    if args.approx_softmax != "adasoftmax":
                        indiv_loss = un_crit(output_flat, targets)
                    else:
                        indiv_loss = criterion.forward_all(output, targets)
                    """make sure these are reshaped correctly,
                    because columns are batch len instead of seq length"""
                    indiv_loss = indiv_loss.view(output.size(0), output.size(1))
                elif save_losses and ecoc_nbits is not None:
                    print("still deciding what loss to use for ecoc")
                    indiv_loss = un_crit(output_flat, targets)

                total_loss += len(data) * loss.item()
                hidden = repackage_hidden(hidden)

                if save_losses:
                    loss_per_tstep.append(indiv_loss)

        if args.save_test_batch:
            test_batcher['batch'] = torch.cat(test_batch).cpu().numpy()
            test_batcher['vocab'] = corpus.dictionary
            save_path = "C:/Users/jamesoneill/Projects/NLP/GOLM/golm/" \
                        "golm/golm_hil/results/ptb/test_batch/test_batch.pkl"
            print(save_path)
            save_vocab(test_batcher, save_path)

        if save_losses:
            losses = torch.cat(loss_per_tstep).cpu().numpy()
            save_path = args.results_path.replace("dropout", "losses").replace(".pickle","")
            """ AttributeError: 'numpy.ndarray' object has no attribute 'write' """
            np.save(save_path, losses)

        nbatches = len(data_source)
        measures = total_loss / nbatches
        if args.codebook or args.loss == "bce":
            val_ham = total_ham / nbatches
            val_acc = total_acc / nbatches
            measures = [val_loss, val_acc, val_ham]
        if args.accuracy: measures = [val_loss, total_acc/ nbatches]
        if args.wmd:
            performance[data_split+'_wmd_diversity'].append(total_d_wmd/nbatches)
            performance[data_split+'_wmd_quality'].append(total_q_wmd/nbatches)
            # measures += [total_q_wmd/nbatches, total_d_wmd/nbatches]
        if args.bleu:
            performance[data_split+'_bleu_diversity'].append(total_d_bleu/nbatches)
            performance[data_split+'_bleu_quality'].append(total_q_bleu/nbatches)
            # measures += [total_q_bleu/nbatches, total_d_bleu/nbatches]
        return measures

    def train(train_perc=None):

        # Turn on training mode which enables dropout.
        model.train()

        total_loss = 0.
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(args.batch_size) # (2, 20, 200)

        if args.codebook or args.loss == "bce" or args.accuracy:
            total_acc = 0.
            total_ham = 0.

        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):

            data, targets = get_batch(train_data, i, args.bptt)
            hidden = repackage_hidden(hidden)

            if args.loss == 'bce':
                # was originally if ecoc_nbits is not None but was changed
                # because ecoc_nbits is not None when args.codebook=True and
                # args.loss !=bce which we don't need in that case
                targ_ind = targets.clone()
                targets = corpus.error_codes[targets.flatten()]
                # print("Error coding being used, targets changed to size {}".format(targets.shape))

            """Don't change if conditions here, only needed
             for curriculum or ss where ss_prob is updated.
             I just needed to turn off scheduled sampler in dropout test """

            if args.scheduled_sampler is not None \
                    and args.dropout_method not in 'curriculum':
                if args.cw_mix:
                    output, hidden = model.forward_ss(data, hidden, target=targets, p=args.ss_prob)
                else:
                    # print(args.scheduled_sampler)
                    inds = list(range(data.size(0)))
                    # less than because the probability is increasing with number of batches as far as uthresh
                    sampled_inds = [ind for ind in inds if random.uniform(0, 1) < args.ss_prob]
                    # use scheduled sampling for generator so minimize the difference between that and
                    # the model that does not use its own predictions.
                    output, hidden = model.forward_ss(data, hidden, inds=sampled_inds)

                # if it changes during training
                if args.scheduled_sampler != 'static':
                    args.ss_prob = update_ss_prob(train_perc, args.scheduled_sampler, args.ss_uthresh)

            elif args.dropout_method in 'curriculum':
                output, hidden = model.forward(data, hidden, args.ss_prob)
                args.ss_prob = update_ss_prob(train_perc, args.scheduled_sampler, args.dropout_ub)

            else:
                """ony works with sgd, provide fix for adam and others"""
                if args.approx_softmax == "hsoftmax" or\
                        args.approx_softmax == "adasoftmax":
                    output, hidden = model(data, hidden, target=targets)
                elif args.approx_softmax == "sampled_softmax":
                    output, hidden, new_targets = model(data, hidden, target=targets)
                else:
                    output, hidden = model(data, hidden)

            if args.neighbor_sampler:
                if args.scheduled_sampler != 'static':
                    args.ns_prob = update_ss_prob(train_perc,
                                                  args.scheduled_sampler, args.ns_uthresh)
                # this is a TPRS function
                targets = corpus.sample_neighbor_sequence(targets, args.ns_prob)

            if args.loss != 'bce':
                if args.codebook:
                    yhat = output.view(output.size(0)*output.size(1), -1)
                    loss = criterion(yhat, targets)
                    yhat_val, yhat_ind = yhat.max(1)
                    predicted_codes, target_codes = corpus.codebook.\
                        lookup_codes(corpus.error_codes, yhat_ind, targets)
                    ham_dist = hamming_distance(predicted_codes, target_codes)
                    # might need - cacc.get_code_accuracy(predicted_codes, targ_ind)
                    accuracy = get_accuracy(yhat_ind, targets)
                else:
                    if args.approx_softmax == "hsoftmax":
                        loss = -torch.mean(torch.log(output))
                    elif args.approx_softmax == "adasoftmax":
                        loss = criterion(output, targets)
                    elif args.approx_softmax == "nce":
                        loss = model.nce_loss(output, targets.view(output.size(0), output.size(1)))
                        """
                          loss = criterion(prob_model, prob_noise_in_model,
                                         prob_noise, prob_target_in_noise)
                        """
                    elif args.approx_softmax == "sampled_softmax":
                        loss = criterion(output, new_targets)
                    else:
                        loss = criterion(output.view(-1, ntokens), targets)

                    if args.accuracy:
                        # should fix for "adasoftmax"
                        if args.approx_softmax in ["hsoftmax", "hsoft_mix",
                                                   "hsoft_mix_tuned", "nce"]:
                            if args.approx_softmax == "nce":
                                targets = targets.view(output.size())
                            accuracy = get_accuracy(output, targets)
                        else:
                            yhat = torch.topk(output, 1)[1]
                            accuracy = get_accuracy(yhat, targets)
                        total_acc += accuracy
            else:
                yhat = output.view(output.size(0)*output.size(1), -1)
                loss = criterion(yhat, targets)
                # if args.codebook:
                # no need for a conditional, if args.loss == bce then
                # codebook is assumed
                # targets already codes here, so don't think i actually need this below
                """
                _, target_codes = corpus.codebook.\
                        lookup_codes(corpus.error_codes, yhat, targets)
                """

                predicted_codes = (yhat > t).float() * 1
                ham_dist = hamming_distance(predicted_codes, targets)
                # use to be targets (codes) with old version of function
                accuracy = cacc.get_code_accuracy(predicted_codes, targ_ind)
                # print(accuracy)

            if args.codebook or args.loss == 'bce':
                performance['train_hamming'].append(ham_dist)
                performance['train_acc'].append(accuracy)

            if args.dropout_method == 'variational':
                loss += KL(model)
            if args.dropout_method == 'concrete':
                """heavily regularized using 10"""
                # print(model.reg_loss)
                loss += model.reg_loss[0] * 10

            if args.loss == "wmd_ce":
                """should perform wmd between predicted and target embeddings here"""

            optim.zero_grad()
            loss.backward()

            if 'rce' in args.loss and 'alt' not in args.loss:
                loss = eval_criterion(output.view(-1, ntokens), targets)

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if args.clip is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            if args.optimizer is not None:
                optim.step()
            else:
                if args.pretrained is not None:
                    for name, p in model.named_parameters():
                        if name is not 'encoder.weight':
                            p.data.add_(-lr, p.grad.data)
                else:
                    for name, p in model.named_parameters():
                        """
                        condition purely for adaptive softmax self.tail.1.0/1
                        not sure if needs to be fixed or not
                        """
                        if p.grad is None and args.approx_softmax == "adasoftmax":
                            continue
                        p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()

            if args.loss == "bce" or args.codebook:
                total_acc += accuracy
                total_ham += ham_dist
            elif args.accuracy: total_acc += accuracy

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                if args.loss == "bce" or args.codebook:
                    acc_av = total_acc / args.log_interval
                    ham_av = total_ham / args.log_interval
                    av_str = " | {:5.2f} accuracy | {:5.2f} hamming distance".format(acc_av, ham_av)
                    performance['train_hamming'].append(ham_av)
                    performance['train_acc'].append(acc_av)
                    total_acc = 0.
                    total_ham = 0.
                elif args.accuracy:
                    acc_av = total_acc / args.log_interval
                    av_str = " | {:5.2f} accuracy ".format(acc_av)
                    performance['train_acc'].append(acc_av)
                    total_acc = 0.

                elapsed = time.time() - start_time
                ppl = math.exp(cur_loss)
                measure = "ppl" if ecoc_nbits is None else "ll"

                r_out = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} ' \
                        '| loss {:5.2f} | {} {:8.3f} '.format(epoch, batch, len(train_data) // args.bptt,
                                            lr, elapsed * 1000 / args.log_interval, cur_loss, measure, ppl)
                if args.bleu or args.wmd:
                    output_val, output_ind = output.max(1)
                    out_b = output_ind.view(data.size()).t().cpu().numpy()
                    targ_b = targets.view(data.size()).t().cpu().numpy()
                    out_b_new = out_b.copy()
                    np.random.shuffle(out_b_new)
                    if args.bleu:
                        bleu_q = get_bleu(out_b, targ_b)
                        bleu_d = get_bleu(out_b, out_b_new)
                        r_out += "| bleu q {:.2f} | bleu d {:.2f} ".format(bleu_q, bleu_d)
                        performance['train_bleu_diversity'].append(bleu_d)
                        performance['train_bleu_quality'].append(bleu_q)
                    if args.wmd:
                        wmd_d = get_wmd(out_b, targ_b, corpus, pre_emb)
                        wmd_q = get_wmd(out_b, out_b_new, corpus, pre_emb)
                        r_out += "| wmd q {:.2f} | wmd d {:.2f} ".format(wmd_q, wmd_d)
                        performance['train_wmd_diversity'].append(wmd_d)
                        performance['train_wmd_quality'].append(wmd_q)

                print(r_out)

                if args.loss == "bce" or args.codebook:
                    print("\t\t\t {}".format(av_str))

                total_loss = 0
                start_time = time.time()
                performance['train_epoch'].append(epoch)
                performance['train_lr'].append(lr)
                performance['train_loss'].append(cur_loss)
                performance['train_ppl'].append(ppl)

                if args.dropout_method == 'concrete':
                    # show_drop_probs(model, args.dropout_position)
                    if args.dropout_position == 1 or args.dropout_position == 3:
                        performance['drop_in_rate'].append(float(model.drop_in.p[0]))
                    if args.dropout_position == 2 or args.dropout_position == 3:
                        performance['drop_out_rate'].append(float(model.drop_out.p[0]))

            # print("Feedforward Time: {} ".format(time.clock() - start))
        # its actually the results

    def export_onnx(path, batch_size, seq_len):

        print('The model is also exported in ONNX format at {}'.
              format(os.path.realpath(args.onnx_export)))
        model.eval()
        dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
        hidden = model.init_hidden(batch_size)
        torch.onnx.export(model, (dummy_input, hidden), path)

    # Loop over epochs.
    lr = args.lr  # if args.optimizer == None:
    best_val_loss = None
    anneal_inc = 0
    anneal_dec = False

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_perc = epoch / float(args.epochs)
            train(train_perc)
            val_loss = evaluate(val_data)

            performance = update_performance(val_loss, performance, test=False)
            performance['val_epoch'].append(epoch)
            performance['val_lr'].append(lr)
            performance['val_loss'].append(val_loss)

            if best_val_loss is None or val_loss < best_val_loss:
                if args.save_model:
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                best_val_loss = val_loss
                # if the learning rate has been decreased and we see continuous
                #  improvements 2 epochs after we boost it back up
                if args.control and anneal_inc % 2 == 0 and anneal_dec == True:
                    lr *= 2.0
                    anneal_dec = False
                anneal_inc += 1
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                # this is not good when using nnrs because as the initial noise goes up scores are slightly worse
                # and the annealing is to severe
                # perhaps have a counter if the performance is improving after annealing to raise it higher
                if args.control:
                    anneal_dec = True
                lr /= args.decay_factor

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    if args.save_model:
        # Load the best saved model.
        with open(args.save, 'rb') as f:
            model = torch.load(f)
            # after load the recurrent params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward pass
            model.rnn.flatten_parameters()

    # Run on test data.

    test_loss = evaluate(test_data, args.save_losses, data_split="test")
    performance = update_performance(test_loss, performance, test=True)

    save_vocab(performance, args.results_path, show_len=False)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)


if __name__ == "__main__":
    run()
