import math
import numpy as np


def old_update_ss_prob(ss_prob, sampler, i, train_len):
    if sampler == 'fixed':
        # no need to change ss_prob when fixed
        pass
    elif sampler == 'linear_decay':
        ss_prob -= (i / train_len)
    elif sampler == 'exp_decay':
        ss_prob -= np.exp(-i / train_len)
    elif sampler == 'sigmoid_decay':
        # add weights here
        ss_prob -= 1.0 / 1.0 + np.exp(-i / train_len)
    assert ss_prob > 0
    return ss_prob


def e(x):
    return 1 - (1-x) * np.exp(-x)


def e1(x):
    return 1-(1-x)/(x + np.exp(-x))


def sig_above(x):
    return 1 - 2/(np.exp(10*x)+1)


def sig_below(x):
    x = 1-x
    return 2.0/(np.exp(10*x)+1)


def adjustable_s_curve(x):
    return 0.5 * (1 + np.sin((x * math.pi)-(math.pi/2)))


# uthresh = upper threshold on sample probability

def update_ss_prob(x,  decay='linear', uthresh=1.0):
    if decay == 'linear':
        ss_prob = x
    elif decay == 'fast' or decay == 'exp':
        ss_prob = e(x)
    elif decay == 'very fast':
        ss_prob = sig_above(x)
    elif decay == 'very slow':
        ss_prob = sig_below(x)
    elif decay == 'nearly_linear':
        ss_prob = e1(x)
    elif decay == 'sigmoid':
        ss_prob = adjustable_s_curve(x)
    assert ss_prob <= 1.0
    if ss_prob > uthresh:
        ss_prob = uthresh
    return ss_prob





