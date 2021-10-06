""" Tests the Latent Neural Language Model """

# coding: utf-8

import torch.onnx
from models.recurrent.rnn import RNNModel
from loaders.helpers import save_vocab
from utils.metrics.dist import get_predict_token_vector, get_nearest_token
from models.loss.loss import get_criterion
from models.optimizer import get_optimizer, get_scheduler
from utils.batchers import *


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
        args.nhid = args.nhid

    # keys are ordered so its ok to use values here
    # print(torch_vocab_vectors.keys())
    torch_vocab_vectors = torch.cuda.FloatTensor(list(corpus.dictionary.wv.values()))

    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, args.eval_batch_size, device)
    test_data = batchify(corpus.test, args.eval_batch_size, device)
    ntokens = len(corpus.dictionary)

    print("Number of Tokens: {}".format(ntokens))

    embeddings = corpus.dictionary.wv if args.pretrained is not None else None
    model = RNNModel(rnn_type=args.model, ntoken=ntokens, ninp=args.emsize, nhid=args.nhid,
                     nlayers=args.nlayers, pretrained=embeddings, dropout=args.dropout,
                     pos=args.pos, tie_weights=args.tied, latent=args.latent).to(device)

    criterion = get_criterion(args.loss)
    optim = get_optimizer(model, args.optimizer, args.lr)

    # if None, do the original annealing from lr = 20.
    if args.scheduler is not None and args.optimizer is not None:
        scheduler = get_scheduler(optim, args, train_size=train_data.size(0))

    measure = 'acc'
    performance = {'train_epoch': [], 'train_loss': [], 'train_' + measure: [], 'train_lr': [],
                   'val_epoch': [], 'val_loss': [], 'val_' + measure: [], 'val_lr': [],
                   'test_loss': [], 'test_' + measure: []}

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()

        total_loss = 0.
        total_acc = 0.

        hidden = model.init_hidden(args.eval_batch_size)

        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, i, args.bptt)
                output, hidden = model(data, hidden)
                try:
                    # probably do not need to flatten using this but not sure
                    output_flat = output.view(-1, args.nhid)
                    # num_samps always 1 for evaluations
                    # was originally passing target_vectors as 2nd argument but that is wrong becuase
                    # its only retrieving prediction vector for that available in the batch and not the
                    # prediction vectors that is most similar to any token in the vocab
                    pred_vector, pred_inds = get_predict_token_vector(output_flat, torch_vocab_vectors, s=1)
                    # now get associated vector from target indices

                    targets = targets.flatten()
                    target_vector = torch_vocab_vectors[targets]

                    loss = criterion(pred_vector, target_vector)
                    predicted = pred_inds.data
                    total_loss += len(data) * loss.item()
                    total_acc += (predicted == targets).sum().item()

                    hidden = repackage_hidden(hidden)
                except:
                    print(output.size())
                    print(targets.size())

        val_ls = total_loss / len(data_source)
        val_ac = total_acc / len(data_source)
        if args.scheduler is not None:
            scheduler.step(val_loss)

        return val_ls, val_ac

    def train(train_perc=None):
        # Turn on training mode which enables dropout.
        model.train()

        total_loss = 0.
        total_correct = 0.
        start_time = time.time()
        hidden = model.init_hidden(args.batch_size)

        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):

            x_batch, targets = get_batch(train_data, i, args.bptt)

            hidden = repackage_hidden(hidden)
            model.zero_grad()
            output, hidden = model(x_batch, hidden)

            # true : print(output.requires_grad)
            output_flat = output.contiguous().view(-1, args.nhid)

            # true : print(output_flat.requires_grad)
            targets = targets.flatten()
            target_vectors = torch_vocab_vectors[targets]

            if args.latent_nn:
                """When chosen, we use nearest neighbor
                    of predicted vector as the prediction"""
                # this gets nearest known vector from predicted
                # was originally passing in the
                pred_vector, pred_inds = get_predict_token_vector(output_flat, torch_vocab_vectors, s=args.latent_k)
                predicted = pred_inds.data
            else:
                pred_vector = output_flat
                predicted = get_nearest_token(output_flat, torch_vocab_vectors).squeeze(1).data

            assert pred_vector.size() == target_vectors.size()
            assert predicted.size() == targets.size()

            # true : print(pred_vector.requires_grad)

            # no gradients: check_gradients(model)
            loss = criterion(pred_vector, target_vectors)
            # no gradients: check_gradients(model)

            total_correct += (predicted == targets).sum().item() / float(len(targets))
            total_loss += loss.item()

            if args.optimizer is not None:
                optim.zero_grad()

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            if args.optimizer is not None:
                optim.step()
            else:
                # sgd_update(args.pretrained, model.named_parameters(), lr)
                if args.pretrained is not None:
                    for name, p in model.named_parameters():
                        if name is not 'encoder.weight':
                            p.data.add_(-lr, p.grad.data)
                else:
                    for p in model.named_parameters():
                        p.data.add_(-lr, p.grad.data)

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                cur_acc = (total_correct / args.log_interval) * 100

                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'mse {:5.6f} | acc {:8.2f}'.format(epoch, batch, len(train_data) // args.bptt, lr,
                                                          elapsed * 1000 / args.log_interval, cur_loss * 100, cur_acc))
                total_loss = 0
                total_correct = 0
                start_time = time.time()
                performance['train_epoch'].append(epoch)
                performance['train_lr'].append(lr)
                performance['train_loss'].append(cur_loss)
                performance['train_acc'].append(cur_acc)

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
            val_loss, val_acc = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid mse {:5.6f} | '
                  'valid acc {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss * 100, val_acc))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.

            performance['val_epoch'].append(epoch)
            performance['val_lr'].append(lr)
            performance['val_loss'].append(val_loss)
            performance['val_acc'].append(val_acc)

            if best_val_loss is None or val_loss < best_val_loss:
                # with open(args.save, 'wb') as f:
                #    torch.save(model, f)
                best_val_loss = val_loss
                # if the learning rate has been decreased and we see continuous
                #  improvements 2 epochs after we boost it back up
                if args.control and anneal_inc % 2 == 0 and anneal_dec is True:
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
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss, test_acc = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, test_acc))
    print('=' * 89)
    performance['test_loss'].append(test_loss)
    performance['test_ppl'].append(test_acc)

    save_vocab(performance, args.results_path, show_len=False)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
