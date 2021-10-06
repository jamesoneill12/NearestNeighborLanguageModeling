""" Original PyTorch Example - Primarily because adam would not work in lm_train """


# coding: utf-8
import torch.onnx
from models.networks.recurrent import RNNModel
from loaders.helpers import save_vocab
from trainers.lm.train_helpers import update_performance
from utils.eval.bleu import get_bleu
from models.optimizer import get_optimizer, get_scheduler, ScheduledOptim
from utils.batchers import *


# target corpus is now added in the case of seq2seq
def run(args, corpus, dec_corpus=None, mod_config=None):
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

    ecoc_nbits = None
    freq = None


    # codebook=args.codebook,
    if 'transformer' not in args.rnn_type.lower():
        model = RNNModel(rnn_type=args.model, ntoken=ntokens, ninp=args.emsize, nhid=args.nhid, approx_softmax=args.approx_softmax,
                     noise_ratio=args.noise_ratio, norm_term=args.norm_term, nlayers=args.nlayers, ss_emb=args.ss_emb,
                     cw_mix=args.cw_mix, bsize=args.batch_size, drop_rate=args.dropout, unigram_dist=freq,pretrained=None,
                     drop_method=args.dropout_method, drop_position=args.dropout_position,  ecoc_nbits=ecoc_nbits, vocab_size=ntokens,
                     fixed_drop=args.fixed_dropout, dropc=args.dropconnect, pos=args.pos, nud_tags=nud_tags, nptb_tags=nptb_tags,
                     batch_norm=args.batch_norm, tie_weights=args.tied).to(device)
    else:
        # num_enc_heads = 3, enc_nhid = None, total_enc_key_depth = None,
        # total_enc_value_depth = None, enc_filter_size = 10, vocab_size = None,
        # num_dec_heads = 4, dec_nhid = None, total_dec_key_depth = None,
        # total_dec_value_depth = None, filter_size = 10, model_args = None
        # rnn_type=args.model, ninp=args.emsize, nhid=args.nhid, dropout=args.dropout_method,
        #                         vocab=ntokens, max_len=100, in_drop=args.dropout, hid_dropr=args.dropout,
        #                         att_dropr=args.dropout, nlayers=args.nlayers, drop_rate=args.dropout,
        #                         vocab_size=ntokens
        from models.networks.transformer.transformer import get_transformer
        if mod_config is not None:
            mod_config.n_embed = args.emsize
            mod_config.vocab_size = ntokens
            model = get_transformer(tf_type=args.model, args=mod_config).cuda()

    #optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.optimizer == 'adam_w_r':
        optim = ScheduledOptim(
            torch.optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            mod_config.d_model, mod_config.n_warmup_steps)
    else:
        optim = get_optimizer(model.parameters(), args.optimizer, args.lr)

    if args.scheduler is not None: scheduler = get_scheduler(args, optim, train_data)
    criterion = nn.CrossEntropyLoss()

    ###############################################################################
    # Training code
    ###############################################################################

    measure = 'ppl' if 'pos' not in args.data else 'acc'
    performance = {'train_epoch': [], 'train_loss': [], 'train_' + measure: [], 'train_lr': [],
                   'val_epoch': [], 'val_loss': [], 'val_' + measure: [], 'val_lr': [],
                   'test_loss': [], 'test_' + measure: []}

    """ uncertainty saves the calibrated output probabilities """
    def evaluate(data_source, save_losses=False):

        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        total_quality_bleu = 0.
        total_diversity_bleu = 0.
        num_tokens = len(corpus.dictionary)
        if 'transformer' not in args.rnn_type.lower():
            hidden = model.init_hidden(eval_batch_size)
        else:
            hidden = None
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, i, args.bptt)
                if 'transformer' not in args.rnn_type.lower():
                    output, hidden = model(data, hidden)
                else:
                    output = model(data)
                yhat = output.view(-1, ntokens)
                loss = criterion(yhat, targets)
                total_loss += len(data) * loss.item()
                #total_quality_bleu += get_bleu(output, targets.view(output.size()))  # average bleu between different preds in batch
                #total_diversity_bleu += get_bleu(output, targets.view(output.size()))  # bleu with target and pred
                if 'transformer' not in args.rnn_type.lower():
                    hidden = repackage_hidden(hidden)
        nbatches = len(data_source)
        val_loss = total_loss / nbatches
        #val_quality_bleu = total_quality_bleu / nbatches
        #val_diversity_bleu = total_loss / nbatches
        return val_loss

    def train(train_perc=None):

        # Turn on training mode which enables dropout.
        model.train()

        total_loss, total_bleu_quality, total_bleu_diversity = 0., 0., 0.
        start_time = time.time()
        ntokens = len(corpus.dictionary)

        if 'transformer' not in args.rnn_type.lower():
            hidden = model.init_hidden(args.batch_size) # (2, 20, 200)
        else:
            hidden = None

        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):

            data, targets = get_batch(train_data, i, args.bptt)
            optim.zero_grad()

            if 'transformer' not in args.rnn_type.lower():
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            else:
                output = model(data)

            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if args.clip is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            if args.optimizer is not None:
                optim.step()
            else:
                for name, p in model.named_parameters():
                    if p.grad is None: continue
                    p.data.add_(-lr, p.grad.data)

            if args.optimizer == "adamw" and args.scheduler == "cosine_restarts":
                scheduler.batch_step()

            total_loss += loss.item()
            if 'transformer' in args.model.lower():
                output_val, output_ind = output.max(2)
            else:
                output_val, output_ind = output.max(1)
            out_f = output_ind.view(data.size()).t().cpu().numpy()
            out_s = targets.view(data.size()).t().cpu().numpy()
            out_f_new = out_f.copy()
            total_bleu_quality += get_bleu(out_f, out_s)
            np.random.shuffle(out_f_new)
            total_bleu_diversity += get_bleu(out_f, out_f_new)

            if batch % 20 == 0:
                cur_loss = total_loss/20
                elapsed = time.time() - start_time
                ppl = math.exp(cur_loss)
                measure = "ppl"
                learn_rate = optim.defaults['lr'] if args.optimizer == "adamw" else args.lr
                cur_bleu_quality = total_bleu_quality/20
                cur_bleu_diversity = total_bleu_diversity/20
                #,seq_len=output.size(0), bleu_len=4)  # average bleu between different preds in batch
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | {} {:8.3f} | bleu diversity {:.2f} | bleu quality {:.2f}'.format(epoch, batch,
                   len(train_data) // args.bptt, learn_rate, elapsed * 1000 / args.log_interval, cur_loss,
                   measure, ppl, cur_bleu_diversity, cur_bleu_quality))
                total_loss = 0.
                total_bleu_diversity = 0.
                total_bleu_quality = 0.


    def export_onnx(path, batch_size, seq_len):

        print('The model is also exported in ONNX format at {}'.
              format(os.path.realpath(args.onnx_export)))
        model.eval()
        dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)

        if 'transformer' not in args.rnn_type.lower():
            hidden = model.init_hidden(batch_size)
        else:
            hidden = None
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
                if args.control and anneal_inc % 2 == 0 and anneal_dec == True:
                    lr *= 2.0
                    anneal_dec = False
                anneal_inc += 1
            else:
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

    test_loss = evaluate(test_data, args.save_losses)
    performance = update_performance(test_loss, performance, test=True)

    save_vocab(performance, args.results_path, show_len=False)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)


if __name__ == "__main__":
    from settings import *
    from loaders import data
    from configs.models.transformer import get_trans_config

    args.data = PTB_ROOT
    args.model = "TRANSFORMER_LM"
    args.rnn_type = args.model
    vhiers = False
    args.nsprob = 0.0 # starting probability
    args.ss_thresh = None
    args.optimizer = 'adam' # None, amsgrad, sgd with lr annealing
    args.scheduler = None # 'cosine_restarts'
    args.scheduler = None
    args.save_model = False
    args.epochs = 20
    args.lr = 1e-4
    args.control = True
    corpus = data.Corpus(args.data, emb_type=args.pretrained, limit=args.vocab_limit)
    mod_config = get_trans_config()
    run(args, corpus, mod_config=mod_config)
