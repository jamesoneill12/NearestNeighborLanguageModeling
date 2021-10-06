# coding: utf-8

import time
import os
import torch.onnx
from loaders.helpers import save_vocab
from models.posteriors.hsoftmax import HierarchicalSoftmax
from models.posteriors.misc import Word2VecEncoder
from models.samplers.ss import update_ss_prob
from models.recurrent.rnn import RNNModel
from utils.batchers import *
from models.loss.loss import get_criterion
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

    train_data = corpus.train
    val_data = corpus.valid
    test_data = corpus.test
    ntokens = len(corpus.dictionary)
    nud_tags = len(corpus.ud_vocab)
    nptb_tags = len(corpus.ptb_vocab)
    nptb_tags = nptb_tags if args.nptb_token else None
    print("Number of ptb tags: {0} \t ud tags {1}:".format(nptb_tags, nud_tags))

    if args.adversarial:
        embeddings = corpus.dictionary.wv if args.pretrained is not None else None
        model_gen = RNNModel(rnn_type=args.model, ntoken=ntokens, ninp=args.emsize, nhid=args.nhid,
                                    nlayers=args.nlayers, pretrained=embeddings, dropout=args.dropout,
                                    pos=args.pos, nud_tags=nud_tags, nptb_tags=nptb_tags,
                                    tie_weights=args.tied).to(device)
        model_disc = RNNModel(rnn_type=args.model, ntoken=ntokens, ninp=args.emsize, nhid=args.nhid,
                                     nlayers=args.nlayers, pretrained=embeddings, dropout=args.dropout,
                                     pos=args.pos, nud_tags=nud_tags,
                                     nptb_tags=nptb_tags, tie_weights=args.tied).to(device)
        adv_criterion = get_criterion(args.adv_loss)
        optimD = torch.optim.Adam(model_disc.parameters(), lr=args.disc_lr, betas=(args.b1, args.b2))
        optimG = torch.optim.Adam(model_gen.parameters(), lr=args.gen_lr, betas=(args.b1, args.b2))
    else:
        embeddings = corpus.dictionary.wv if args.pretrained is not None else None
        model = RNNModel(rnn_type=args.model, ntoken=ntokens, ninp=args.emsize, nhid=args.nhid,
                                nlayers=args.nlayers, pretrained=embeddings, dropout=args.dropout,
                                pos=args.pos, nud_tags=nud_tags, nptb_tags=nptb_tags,
                                tie_weights=args.tied).to(device)

    criterion = get_criterion(args.loss)

    if args.hsoftmax:
        encoder = Word2VecEncoder(ntokens, args.emsize, args.dropout)
        hierarchical_softmax = HierarchicalSoftmax(ntokens, args.nhid)
        model.add_module("encoder", encoder)
        model.add_module("decoder", hierarchical_softmax)

    elif not args.adversarial:
        if args.optimizer == 'amsgrad':
            optim = torch.optim.Adam(model.parameters(), amsgrad=True)
        elif args.optimizer == 'sgdr':
            optim = torch.optim.Adam(model.parameters(), amsgrad=True)
            # FINISH !
            # optimizer = torch.optim.CosineA
        elif args.optimizer is not None:
            optim = torch.optim.Adam(model.parameters())

    # if None, do the original annealing from lr = 20.
    if args.optimizer is not None:
        if args.scheduler == 'cosine_anneal':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20000, eta_min=0, last_epoch=-1)
        elif args.scheduler == 'lro':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min')
        elif args.scheduler == 'multi_step':
            max_epochs = args.epochs * train_data.size(0) / args.bptt
            mstones = list(range(0, max_epochs, max_epochs / 10))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=mstones, gamma=0.1)

    ###############################################################################
    # Training code
    ###############################################################################

    measure = 'ppl' if 'pos' not in args.data else 'acc'
    performance = {'train_epoch': [], 'train_loss': [], 'train_' + measure: [], 'train_lr': [],
                   'val_epoch': [], 'val_loss': [], 'val_' + measure: [], 'val_lr': [],
                   'test_loss': [], 'test_' + measure: []}


    def evaluate_pos(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        total_acc = 0.
        ntokens = len(corpus.dictionary.idx2word)
        # was eval batch size but changed because thought it was causing error
        hidden = model.init_hidden(args.batch_size)
        with torch.no_grad():
            for i, batch in enumerate(iter(data_source)):
                data, targets = batch.word, batch.udtag
                output, hidden = model(data, hidden)
                if args.hsoftmax:
                    probs = hierarchical_softmax(output.view(-1, output.size(2)), targets)
                    loss = -torch.mean(torch.log(probs))
                else:
                    output_flat = output.view(-1, nud_tags)
                    loss = criterion(output_flat, targets)
                    _, predicted = torch.max(output_flat.data, 1)
                hidden = repackage_hidden(hidden)
                total_loss += len(data) * loss.item()
                total_acc += (predicted == targets).sum().item()
        val_acc = 100 * (total_acc / len(data_source))
        val_loss = total_loss / len(data_source)
        if args.scheduler is not None:
            scheduler.step(val_loss)
        return val_loss, val_acc

    def train_pos(train_perc=None):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.
        total_correct = 0.
        start_time = time.time()
        ntokens = len(corpus.dictionary.idx2word)
        hidden = model.init_hidden(args.batch_size)
        for i, batch in enumerate(iter(train_data)):
            x_train, targets = batch.word, batch.udtag
            if x_train.size(1) != args.batch_size:
                pad_tensor = torch.zeros((x_train.size(0), args.batch_size - x_train.size(1))).type(torch.cuda.LongTensor)
                x_train = torch.cat([x_train, pad_tensor], 1)
                targets = torch.cat([targets, pad_tensor], 1)

            hidden = repackage_hidden(hidden)
            model.zero_grad()

            if args.scheduled_sampler is not None:
                inds = list(range(x_train.size(0)))
                sampled_inds = [ind for ind in inds if random.uniform(0, 1) < args.ss_prob]
                if args.nptb_token:
                    ptb_output, output, hidden = model.forward_ss(x_train, hidden, sampled_inds)
                else:
                    output, hidden = model.forward_ss(x_train, hidden, sampled_inds)
                if args.scheduled_sampler != 'static':
                    args.ss_prob = update_ss_prob(train_perc, args.scheduled_sampler, args.ss_uthresh)
            else:
                if args.nptb_token:
                    ptb_output, output, hidden = model(x_train, hidden)
                else:
                    output, hidden = model(x_train, hidden)

            if args.neighbor_sampler:
                if args.scheduled_sampler != 'static':
                    args.ns_prob = update_ss_prob(train_perc, args.scheduled_sampler, args.ns_uthresh)
                targets = corpus.sample_neighbor_sequence(targets, args.ns_prob)

            if args.hsoftmax:
                probs = hierarchical_softmax(output.view(-1, output.size(2)), targets)
                loss = -torch.mean(torch.log(probs))
            else:
                # backward wont work for ptb_/output.permute(0,2,1).contiguous() because shouldn't contiguous()
                output = output.view(-1, output.size(2))
                targets = targets.view(-1, )
                loss = criterion(output, targets)
                if args.nptb_token:
                    ptb_output = ptb_output.view(-1, output.size(2))
                    ptb_loss = criterion(ptb_output, targets)
                    loss = (loss + ptb_loss) / 2.0

            if args.optimizer is not None:
                optim.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if args.optimizer is not None:
                optim.step()
            else:
                if args.pretrained is not None:
                    for name, p in model.named_parameters():
                        if name is not 'encoder.weight':
                            p.data.add_(-lr, p.grad.data)
                else:
                    for p in model.parameters():
                        p.data.add_(-lr, p.grad.data)
                if args.adversarial:
                    if args.pretrained is not None:
                        for name, p in model_gen.named_parameters():
                            if name is not 'encoder.weight':
                                p.data.add_(-lr, p.grad.data)
                    else:
                        for p in model_gen.parameters():
                            p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()
            output_flat = output.view(-1, nud_tags)
            _, predicted = torch.max(output_flat.data, 1)
            total_correct += (predicted == targets).sum().item() / float(len(targets))

            if i % args.log_interval == 0 and i > 0:
                cur_loss = total_loss / args.log_interval
                cur_acc = (total_correct / args.log_interval) * 100
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | acc {:8.2f}'.format(epoch, i, len(train_data) // args.bptt, lr,
                                                          elapsed * 1000 / args.log_interval, cur_loss, cur_acc))
                total_loss = 0
                total_correct = 0
                start_time = time.time()
                performance['train_epoch'].append(epoch)
                performance['train_lr'].append(lr)
                performance['train_loss'].append(cur_loss)
                performance['train_acc'].append(cur_acc)

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
            train_pos(train_perc) if 'pos' not in args.data else train_pos(train_perc)

            val_loss, val_acc = evaluate_pos(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid acc {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, val_acc))
            print('-' * 89)

            performance['val_epoch'].append(epoch)
            performance['val_lr'].append(lr)
            performance['val_loss'].append(val_loss)
            performance['val_acc'].append(val_acc)

            if best_val_loss is None or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
                if args.control and anneal_inc % 2 == 0 and anneal_dec == True:
                    lr *= 2.0
                    anneal_dec = False
                anneal_inc += 1
            else:
                if args.control:
                    anneal_dec = True
                lr /= 4.0

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load the recurrent params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss, test_acc = evaluate_pos(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test acc {:8.2f}'.format(test_loss, test_acc))
    print('=' * 89)
    performance['test_loss'].append(test_loss)
    performance['test_acc'].append(test_acc)

    save_vocab(performance, args.results_path, show_len=False)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)

