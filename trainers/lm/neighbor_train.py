"""
Instead of predicting a single target, we predict a whole neighborhood of tokens (default N=10)
as a multi-label problem. (not to be confused with curriculum-based neighborhood sampling)
We can also weight the importance of each prediction based on the cosine distance of the 10 neighbors
from the original target. This neighbors should act as good regularizers and even when poor predictions are made,
at least they are more likely to be semnatically related.
"""

# coding: utf-8

import math
import torch.onnx
from models.recurrent.rnn import RNNModel
from loaders.helpers import save_vocab
from models.posteriors.split_ce import *
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
    eval_batch_size = 10

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
    model = RNNModel(rnn_type=args.model, ntoken=ntokens, ninp=args.emsize, nhid=args.nhid,
                    nlayers=args.nlayers, bsize=args.batch_size, pretrained=embeddings, drop_rate=args.dropout,
                    drop_method=args.dropout_method, drop_position=args.dropout_position,
                     fixed_drop=args.fixed_dropout, dropc=args.dropconnect, pos=args.pos, nud_tags=nud_tags,
                     nptb_tags=nptb_tags, batch_norm=args.batch_norm, tie_weights=args.tied).to(device)

    criterion = get_criterion(args.loss)

    if 'rce' in args.loss:
        class_weights = [1] * ntokens
        vocab_vectors = corpus.dictionary.wv
        criterion = criterion(class_weights=class_weights, batch_size=args.batch_size,
                              lm=True, vocab_vectors=vocab_vectors, temp=10)
        # rce_alt and rce_neighbor need to compute scaled rewards given by cosine similarities
        eval_criterion = criterion if 'alt' in args.loss or 'neighbor' in args.loss else nn.CrossEntropyLoss()

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
    if args.fixed_dropout and args.dropout_method == 'concrete':
        if args.dropout_position == 1 or args.dropout_position == 3:
            performance['drop_in_rate'] = []
        if args.dropout_position == 2 or args.dropout_position == 3:
            performance['drop_out_rate'] = []

    """ uncertainty saves the calibrated output probabilities """
    def evaluate(data_source, save_losses=False):

        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
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
                output, hidden = model(data, hidden)
                output_flat = output.view(output.size(0) * output.size(1), ntokens)

                if args.loss == 'rce':
                    loss = eval_criterion(output_flat, targets) * criterion(output_flat, targets)
                else:
                    loss = criterion(output_flat, targets)

                if save_losses:
                    indiv_loss = un_crit(output_flat, targets)
                    """make sure these are reshaped correctly,
                    because columns are batch len instead of seq length"""
                    indiv_loss = indiv_loss.view(output.size(0), output.size(1))

                total_loss += len(data) * loss.item()
                hidden = repackage_hidden(hidden)

                if save_losses:
                    loss_per_tstep.append(indiv_loss)

        if save_losses:
            losses = torch.cat(loss_per_tstep).cpu().numpy()
            save_path = args.results_path.replace("dropout", "losses").replace(".pickle","")
            """
            AttributeError: 'numpy.ndarray' object has no attribute 'write'
            """
            np.save(save_path, losses)

        val_loss = total_loss / len(data_source)
        return val_loss

    def train(train_perc=None):

        # Turn on training mode which enables dropout.
        model.train()

        total_loss = 0.
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(args.batch_size) # (2, 20, 200)

        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            data, targets = get_batch(train_data, i, args.bptt)
            hidden = repackage_hidden(hidden)

            """Don't change if conditions here, only needed
             for curriculum or ss where ss_prob is updated.
             I just needed to turn off scheduled sampler in dropout test """
            if args.scheduled_sampler is not None \
                    and args.dropout_method not in 'curriculum':
                # print(args.scheduled_sampler)
                inds = list(range(data.size(0)))
                # less than because the probability is increasing with number of batches as far as uthresh
                sampled_inds = [ind for ind in inds if random.uniform(0, 1) < args.ss_prob]
                # use scheduled sampling for generator so minimize the difference between that and
                # the model that does not use its own predictions.
                output, hidden = model.forward_ss(data, hidden, sampled_inds)
                # if it changes during training
                if args.scheduled_sampler != 'static':
                    args.ss_prob = update_ss_prob(train_perc, args.scheduled_sampler, args.ss_uthresh)
            elif args.dropout_method in 'curriculum':
                output, hidden = model.forward(data, hidden, args.ss_prob)
                args.ss_prob = update_ss_prob(train_perc, args.scheduled_sampler, args.dropout_ub)
            else:
                print(data.size())
                print(hidden[0].size(), hidden[1].size())
                output, hidden = model(data, hidden)

            if args.neighbor_sampler:
                if args.scheduled_sampler != 'static':
                    args.ns_prob = update_ss_prob(train_perc,
                                                  args.scheduled_sampler, args.ns_uthresh)
                # this is a TPRS function
                targets = corpus.sample_neighbor_sequence(targets, args.ns_prob)

            loss = criterion(output.view(-1, ntokens), targets)

            if args.dropout_method == 'variational':
                loss += KL(model)
            if args.dropout_method == 'concrete':
                """heavily regularized using 10"""
                # print(model.reg_loss)
                loss += model.reg_loss[0] * 10

            optim.zero_grad()
            loss.backward()

            if 'rce' in args.loss and 'alt' not in args.loss:
                loss = eval_criterion(output.view(-1, ntokens), targets)

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            if args.optimizer is not None:
                optim.step()
            else:
                if args.pretrained is not None:
                    for name, p in model.named_parameters():
                        if name is not 'encoder.weight':
                            p.data.add_(-lr, p.grad.data)
                else:
                    for name, p in model.named_parameters():
                        p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                ppl = math.exp(cur_loss)
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch, len(train_data) // args.bptt, lr,
                                                          elapsed * 1000 / args.log_interval, cur_loss, ppl))
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
            val_ppl = math.exp(val_loss)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, val_ppl))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.

            performance['val_epoch'].append(epoch)
            performance['val_lr'].append(lr)
            performance['val_loss'].append(val_loss)
            performance['val_ppl'].append(val_ppl)

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
                lr /= 4.0

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

    test_loss = evaluate(test_data, args.save_losses)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, test_ppl))
    print('=' * 89)
    performance['test_loss'].append(test_loss)
    performance['test_ppl'].append(test_ppl)
    save_vocab(performance, args.results_path, show_len=False)
