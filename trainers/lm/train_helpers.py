import math


def add_codebook_eval_metrics(performance):
    performance['train_hamming'] = []
    performance['train_acc'] = []
    performance['val_hamming'] = []
    performance['val_acc'] = []
    performance['test_hamming'] = []
    performance['test_acc'] = []
    return performance


def update_performance(loss, performance, test=False):
    data = "test" if test else "val"
    print('=' * 89)
    if type(loss) == list:
        test_ppl = math.exp(loss[0])
        performance[data+'_acc'].append(loss[1])
        # try-except added in case where
        c_out = '| End of training | {0} loss {1:5.2f} |' \
                ' {0} ppl {2:8.2f} | {0} acc {2:5.2f} '.format(data, loss[0], test_ppl,loss[1])

        if len(loss) < 3:
            print(c_out)
        elif len(loss) == 3:
            performance[data+'_hamming'].append(loss[2])
            c_out += '| {0} hamming {3:8.2f}'.format(loss[2])
            print(c_out)
        elif len(loss) == 4:
            print("Implement for when 4 extra params passed")
    else:
        test_ppl = math.exp(loss)
        print('| End of training | {0} loss {1:5.2f} |'
              ' {0} ppl {2:8.2f}'.format(data, loss, test_ppl))
        performance[data+'_loss'].append(loss)
        performance[data+'_ppl'].append(test_ppl)
    print('=' * 89)
    return performance

