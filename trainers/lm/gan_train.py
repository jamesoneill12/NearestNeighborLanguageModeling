# coding: utf-8

import math
import torch.onnx
from models.recurrent.rnn import RNNModel
from loaders.helpers import save_vocab
from models.posteriors.split_ce import *
from models.samplers.ss import update_ss_prob
from models.posteriors.hsoftmax import HierarchicalSoftmax
from models.loss.loss import get_criterion
from models.optimizer import get_optimizer, get_scheduler
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

    eval_batch_size = 10

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

    criterion = get_criterion(args.loss)

    if args.optimizer is not None:
        schedulerD = get_scheduler(args, optimD, train_data)
        schedulerG = get_scheduler(args, optimD, train_data)

    ###############################################################################
    # Training code
    ###############################################################################

    measure = 'ppl' if 'pos' not in args.data else 'acc'
    performance = {'train_epoch': [], 'train_loss': [], 'train_' + measure: [], 'train_lr': [],
                   'val_epoch': [], 'val_loss': [], 'val_' + measure: [], 'val_lr': [],
                   'test_loss': [], 'test_' + measure: []}

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model_gen.eval()
        model_disc.eval()
        total_loss = 0.
        ntokens = len(corpus.dictionary)
        gen_hidden = model_gen.init_hidden(eval_batch_size)
        disc_hidden = model_disc.init_hidden(eval_batch_size)

        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, i, args.bptt)
                gen_output, gen_hidden = model_gen(data, gen_hidden)
                disc_output, disc_hidden = model_gen(data, disc_hidden)

                output_flat = disc_output.view(-1, ntokens)
                loss = criterion(output_flat, targets)
            total_loss += len(data) * loss.item()
            disc_hidden = repackage_hidden(disc_hidden)

        val_loss = total_loss / len(data_source)

        if args.scheduler is not None:
            schedulerD.step(val_loss)
            schedulerG.step(val_loss)
        return val_loss

    def train(train_perc=None):

        # Turn on training mode which enables dropout.
        model_gen.train()
        model_disc.train()

        total_loss = 0.
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        hidden_disc = model_disc.init_hidden(args.batch_size)
        hidden_gen = model_gen.init_hidden(args.batch_size)

        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):

            data, targets = get_batch(train_data, i, args.bptt)
            hidden_disc = repackage_hidden(hidden_disc)
            hidden_gen = repackage_hidden(hidden_gen)

            model_disc.zero_grad()
            model_gen.zero_grad()

            valid = torch.autograd.Variable(torch.Tensor(data.size(0) * data.size(1), 1).fill_(1.0),  requires_grad=False).cuda()
            fake = torch.autograd.Variable(torch.Tensor(data.size(0) * data.size(1), 1).fill_(0.0), requires_grad=False).cuda()

            inds = list(range(data.size(0)))
            if args.scheduled_sampler is not None:
                # less than because the probability is increasing with number of batches as far as uthresh
                sampled_inds = [ind for ind in inds if random.uniform(0, 1) < args.ss_prob]
                output, hidden = model_disc(data, hidden)
                output_gen, hidden_gen = model_gen.forward_ss(data, hidden_gen, sampled_inds)
                if args.scheduled_sampler != 'static':
                    args.ss_prob = update_ss_prob(train_perc, args.scheduled_sampler, args.ss_uthresh)
            else:
                sinds = [ind for ind in inds if random.uniform(0, 1) < args.ss_prob]
                output, hidden = model_disc(data, hidden)
                output_gen, hidden_gen = model_gen.forward_gen(data, hidden_gen, sinds)

            # then past the predicted outputs (output_gen) to z (indices)
            _, z = torch.max(output_gen.data, 1)
            # I should pass the disc hidden here I guess.
            disc_gen_out = model_disc(z, hidden)
            # provides the generator value through disc network
            disc_output = disc_output.view(-1, ntokens)
            # should adv_criterion be binary ce in my case?
            # it is quite hard to generate a sample that completely
            # matches the true target in the first place so what decides
            # a fake z merit tricking the discriminator ?
            g_loss = adv_criterion(disc_gen_out, targets)
            d_loss = adv_criterion(output_gen, targets)
            loss = g_loss.item() + d_loss.item()
            total_loss += loss
            optimG.zero_grad()
            optimD.zero_grad()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model_disc.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(model_gen.parameters(), args.clip)

            if args.optimizer is not None:
                optimG.step()
                optimD.step()
            else:
                check_gradients(model_disc)
                if args.pretrained is not None:
                    for name, p in model_disc.named_parameters():
                        if name is not 'encoder.weight':
                            p.data.add_(-lr, p.grad.data)
                    for name, p in model_gen.named_parameters():
                        if name is not 'encoder.weight':
                            p.data.add_(-lr, p.grad.data)
                else:
                    for p in model_disc.parameters():
                        p.data.add_(-lr, p.grad.data)
                    for p in model_gen.parameters():
                        p.data.add_(-lr, p.grad.data)

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

            # print("Feedforward Time: {} ".format(time.clock() - start))
        # its actually the results

    def export_onnx(path, batch_size, seq_len):

        print('The model is also exported in ONNX format at {}'.
              format(os.path.realpath(args.onnx_export)))
        model_gen.eval()
        model_disc.eval()
        dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)

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
                with open(args.save+"_gen", 'wb') as f:
                    torch.save(model_gen, f)
                with open(args.save+"_disc", 'wb') as f:
                    torch.save(model_disc, f)
                best_val_loss = val_loss
                # if the learning rate has been decreased and we see continuous
                #  improvements 2 epochs after we boost it back up
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

    with open(args.save+"_disc", 'rb') as f:
        model_disc = torch.load(f)
        model_disc.rnn.flatten_parameters()
    with open(args.save+"_gen", 'rb') as f:
        model_gen = torch.load(f)
        model_gen.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, test_ppl))
    print('=' * 89)
    performance['test_loss'].append(test_loss)
    performance['test_ppl'].append(test_ppl)

    save_vocab(performance, args.results_path, show_len=False)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)


if __name__ == "__main__":
    run()
