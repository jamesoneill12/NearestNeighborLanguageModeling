# coding: utf-8

import time
import os
import torch.onnx
import random
from models.enc2dec.seq2seq import Seq2Seq
from loaders.helpers import save_vocab, padded_tensor
from models.loss.loss import get_criterion
from models.samplers.ss import update_ss_prob
from utils.batchers import *
from utils.eval.bleu import get_bleu
from torch.distributions import Categorical
from models.optimizer import get_optimizer

import sys

# target corpus is now added in the case of seq2seq
def run_mt(args, corpus, dec_corpus=None):
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Device: {}".format(device))

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    train_data = corpus.train
    val_data = corpus.valid
    test_data = corpus.test
    # careful that dictionary __len__ gives german vocabulary size
    src_tags = len(corpus.de_vocab)
    trg_tags = len(corpus.en_vocab)

    print("Number of german (source) tokens: {0} \t english (target) tokens {1}:".format(src_tags, trg_tags))

    """
    model = mods.Seq2Seq(svocab_size=src_tags,
                           hidden_size=args.nhid,
                           tvocab_size=trg_tags, attention=args.attention)

    """
    model = Seq2Seq(src_tags, trg_tags, num_layer=args.nlayers,
                    embed_dim=args.nhid, hidden_dim=args.nhid, max_len=30, trg_soi=corpus.en_vocab['<s>'])

    model = model.cuda()
    criterion = get_criterion(args.loss)
    optim = get_optimizer(model.parameters(), args.optimizer, args.lr)

    performance = {'train_epoch': [], 'train_loss': [], 'train_bleu': [], 'train_acc': [], 'train_lr': [],
                   'val_epoch': [], 'val_loss': [], 'val_bleu': [], 'val_acc': [], 'val_lr': [],
                   'test_loss': [], 'test_bleu': [], 'test_acc': []
                   }

    def evaluate_mt(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        total_bleu = 0.
        total_acc = 0.

        """
        Make sure batches are properly assigned from args.batch_size
        """
        with torch.no_grad():
            for i, batch in enumerate(iter(data_source)):
                x_src, x_trg, trg_output = get_mt_batches(batch, corpus)
                if args.attention:
                    output, enc_hidden, dec_hidden, attn_weights = model(x_src, x_trg, enc_hidden, dec_hidden)
                else:
                    output = model(x_src, target=x_trg,  src_length=None)

                target = trg_output.flatten()

                output_flat = output.view(-1, trg_tags)
                loss = criterion(output_flat, target)
                _, predicted = torch.max(output_flat.data, 1)

                # enc_hidden = repackage_hidden(enc_hidden)
                # dec_hidden = repackage_hidden(dec_hidden)

                total_loss += loss.item()
                # total_acc = (predicted == targets).sum().item()
                # think 1st dimension is the sentence length, thats why

                # print("Predicted")
                # print(predicted.size())
                # print(output.size())
                # print(trg_output.size())

                reshape_predicted = predicted.view(trg_output.size())
                # reshape_target = target.view(output.size(0), output.size(1))
                total_bleu += get_bleu(reshape_predicted.cpu().numpy(), trg_output.cpu().numpy())
                total_acc += 100 * (predicted.flatten() ==
                                    target.flatten()).sum().item() / len(predicted.flatten())

        val_size = len(data_source)
        val_acc = total_acc / val_size
        val_bleu = total_bleu / val_size
        val_loss = total_loss / val_size

        return val_loss, val_bleu, val_acc

    def train_mt(train_perc=None):

        # Turn on training mode which enables dropout.
        model.train()
        # model.decoder.train()
        total_loss = 0.
        total_bleu = 0.
        total_acc = 0.
        start_time = time.time()
        # german in this case

        """
        enc_hidden, dec_hidden = model.init_hidden(args.batch_size)
        """

        for i, batch in enumerate(iter(train_data)):

            if args.joint_mt:
                # language model on the source side allows multi-step ahead prediction of context.
                x_src, x_trg, trg_output, src_output = get_seq_batch(x_src, args.joint_mt)
            else:
                x_src, x_trg, trg_output = get_mt_batches(batch, corpus)
                #torch.set_printoptions(threshold=5000)
                #print(x_src.size(), x_trg.size(), trg_output.size())
                #sys.exit()

            batch_size, trg_len = x_trg.size(0), x_trg.size(1)
            # trg_len change in batch_size

            if x_src.size(1) != args.batch_size:
                x_src = padded_tensor(x_src, args.batch_size)

            if x_trg.size(1) != args.batch_size:
                x_trg = padded_tensor(x_trg, args.batch_size)
                # print(x_trg.size()) - torch.Size([39, 56])
                # print(trg_output.size()) - torch.Size([39, 48])
                pad_targs = torch.zeros(trg_output.size(0),
                                        abs(trg_output.size(1) - x_trg.size(1))).type(torch.cuda.LongTensor)
                trg_output = torch.cat([trg_output, pad_targs], 1)

            if args.joint_mt:
                # need to shift indices back one
                x_src = get_batch(x_src, i, args.batch_size)
                y_src = x_src

            # enc_hidden = repackage_hidden(enc_hidden)
            # dec_hidden = repackage_hidden(dec_hidden)

            model.zero_grad()

            if args.scheduled_sampler is not None:

                inds = list(range(x_trg.size(0)))

                if args.sim_mt:
                    sinds = int(args.ss_prob * len(inds))
                    if sinds != 0:
                        sampled_inds = list(reversed(inds))[:sinds]
                else:
                    # using two seperate samples for scheduled sampling.
                    sampled_inds = [ind for ind in inds if random.uniform(0, 1) < args.ss_prob]

                if args.attention:
                    output, enc_hidden, dec_hidden, attn_weights = \
                        model.forward_ss(x_src, x_trg, enc_hidden, dec_hidden, sampled_inds)
                else:
                    output = model.forward_ss(x_src, target=x_trg,  src_length=None)
                    output = output.view(batch_size, x_trg.size(1), -1)

                if args.scheduled_sampler is not 'static':
                    args.ss_prob = update_ss_prob(train_perc, args.scheduled_sampler, args.ss_uthresh)
            else:
                if args.attention:
                    output, enc_hidden, dec_hidden, attn_weights = model(x_src, x_trg, enc_hidden, dec_hidden)
                else:
                    # no need to pass torch.Tensor([src_length]*batch.src.size(1)) as src_length since we used
                    # bucketiterator that efficiently groups sentence of similar length
                    output = model(x_src, target=x_trg,  src_length=None)
                    output = output.view(batch_size, x_trg.size(1), -1)

                    # print(enc_hidden.type())
                    # print(dec_hidden.type())
                    # output, enc_hidden, dec_hidden = model(x_src, x_trg, enc_hidden, dec_hidden)
                    # print(output.type())

            if args.neighbor_sampler:
                if args.scheduled_sampler is not 'static':
                    args.ns_prob = update_ss_prob(train_perc, args.scheduled_sampler, args.ns_uthresh)
                output = dec_corpus.sample_neighbor_sequence(x_trg, args.ns_prob)

            # should be 3-dimensional (bsz, sent_len, ntokens)=(20, 25, 200) but getting (20, 200)
            # output = output.view(-1, output.size(2))
            # x_trg_flat = x_trg.view(-1, 1).squeeze(1)  # de_tags)

            if check_nan(output).item() == 1:
                raise ValueError("Nans in output, be careful with learning rate and gradient clipping.")

            # so problem is that the i'm using english token indices as targets and not the de.

            if args.reinforce:
                """
                How is the sampled action influencing the reward here ?
                """
                m = Categorical(output)
                action = m.sample()
                _, predicted = torch.max(output.data, 1)

                # predict_tensor = predicted.view(x_trg.size(0), -1)
                reshape_predicted = predicted.view(output.size(0), output.size(1))
                reshape_target = target.view(output.size(0), output.size(1))

                reward = get_bleu(reshape_predicted.cpu().numpy(), reshape_target.cpu().numpy())
                loss = - m.log_prob(action) * reward
            else:
                assert output.size(0) == len(trg_output)
                # assert output.size(0) == len(x_trg_flat)
                # print(torch.max(x_trg_flat))
                # loss = criterion(output, x_trg_flat)

                pred = output.contiguous().view(output.size(0) * output.size(1), output.size(2))
                target = trg_output.contiguous().view(-1)

                # print("Prediction {}".format(pred.size()))
                # print("Target {}".format(target.size()))

                loss = criterion(pred, target)

            #  make sure gradients don't explode
            # high learning rates and gradient clipping seem to cause nans
            if (args.clip is not None) and (args.clip > 0) and (args.clip < 2):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                # torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), args.clip)

            # print(loss)
            loss.backward()

            if args.optimizer is not None and args.optimizer != 'sgd':
                optim.step()
                # encoder_optim.step()
                # decoder_optim.step()
            else:
                for name, p in model.named_parameters():
                    # print("{} : Gradient {}".format(name, p.grad is not None))
                    p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()

            # print("Predicted Before")
            # print(output.size())

            _, predicted = torch.max(output.data, 2)
            # total_correct += (predicted == x_trg.sum().item() / float(len(x_trg))

            # print("Output")
            # print(output.size())
            # print("Predicted After")
            # print(predicted.size())

            total_acc += ((predicted.flatten() == target.flatten()).sum().item() / len(predicted.flatten())) * 100

            reshape_predicted = predicted.view(output.size(0),  output.size(1))
            reshape_target = target.view(output.size(0), output.size(1))

            bleu = get_bleu(reshape_predicted.cpu().numpy(), reshape_target.cpu().numpy())
            total_bleu += bleu

            # predict_tensor.size(1))

            # print('| epoch {:3d} | {:5d} batches | bleu {:8.2f}'.format(epoch, i, bleu))

            if i % args.log_interval == 0 and i > 0:
                cur_loss = total_loss / args.log_interval
                cur_bleu = total_bleu / args.log_interval
                cur_acc = total_acc / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:5.2f} | bleu {:8.2f} | acc {:8.2f}'.format(epoch, i, len(train_data) // args.bptt, lr,
                                                                                      elapsed * 1000 / args.log_interval, cur_loss,
                                                                                      math.exp(cur_loss), cur_bleu, cur_acc))
                if args.check_grad:
                    check_gradients(model)

                total_loss, total_bleu, total_acc = 0, 0, 0
                start_time = time.time()
                performance['train_epoch'].append(epoch)
                performance['train_lr'].append(lr)
                performance['train_loss'].append(cur_loss)
                performance['train_bleu'].append(cur_bleu)
                performance['train_acc'].append(cur_acc)

    def export_onnx(path, batch_size, seq_len):

        print('The model is also exported in ONNX format at {}'.
              format(os.path.realpath(args.onnx_export)))
        model.eval()
        dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
        enc_hidden, dec_hidden = model.init_hidden(batch_size)
        torch.onnx.export(model, (dummy_input, enc_hidden, dec_hidden), path)

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
            train_mt(train_perc)

            val_loss, val_bleu, val_acc = evaluate_mt(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid bleu {:8.2f} | valid acc {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, val_bleu, val_acc))
            print('-' * 89)

            performance['val_epoch'].append(epoch)
            performance['val_lr'].append(lr)
            performance['val_loss'].append(val_loss)
            performance['val_bleu'].append(val_bleu)
            performance['val_acc'].append(val_acc)

            if best_val_loss is None or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
                if args.control and anneal_inc % 2 == 0 and anneal_dec:
                    lr *= 2.0
                    anneal_dec = False
                anneal_inc += 1
            elif args.lr_anneal:
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
        # model.recurrent.flatten_parameters()

    # Run on test data.
    test_loss, test_bleu, test_acc = evaluate_mt(test_data)

    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test bleu {:8.2f} | test acc {:8.2f}'.format(test_loss, test_bleu, test_acc))
    print('=' * 89)
    performance['test_loss'].append(test_loss)
    performance['test_bleu'].append(test_bleu)
    performance['test_acc'].append(test_acc)

    save_vocab(performance, args.results_path, show_len=False)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
