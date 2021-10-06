"""
    https://raw.githubusercontent.com/leimao/Sampled_Softmax_PyTorch/master/utils.py
    Sampled Softmax Implementation by Lei Mao
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.samplers.loguniform import LogUniformSampler
import math


class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, nhid, tied_weight):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled

        self.sampler = LogUniformSampler(self.ntokens)
        self.params = nn.Linear(nhid, ntokens)

        if tied_weight is not None:
            self.params.weight = tied_weight
        else:
            in_, out_ = self.params.weight.size()
            stdv = math.sqrt(3. / (in_ + out_))
            self.params.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, labels):
        if self.training:
            # sample ids according to word distribution - Unique
            sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
            logits, new_targets = self.sampled(inputs, labels, sample_values, remove_accidental_match=True)
            return logits, new_targets
        else:
            return self.full(inputs, labels)

    def sampled(self, inputs, labels, sample_values, remove_accidental_match=False):

        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values

        sample_ids = Variable(torch.cuda.LongTensor(sample_ids))
        true_freq = Variable(torch.cuda.FloatTensor(true_freq))
        sample_freq = Variable(torch.cuda.FloatTensor(sample_freq))

        # gather true labels - weights and frequencies
        true_weights = self.params.weight[labels, :]
        true_bias = self.params.bias[labels]

        # gather sample ids - weights and frequencies
        sample_weights = self.params.weight[sample_ids, :]
        sample_bias = self.params.bias[sample_ids]

        # calculate logits
        true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1) + true_bias
        sample_logits = torch.matmul(inputs, torch.t(sample_weights)) + sample_bias
        # remove true labels from sample set
        if remove_accidental_match:
            acc_hits = self.sampler.accidental_match(labels.data.cpu().numpy(), sample_ids.data.cpu().numpy())
            acc_hits = list(zip(*acc_hits))
            sample_logits[acc_hits] = -1e37

        # perform correction
        true_logits = true_logits.sub(torch.log(true_freq))
        sample_logits = sample_logits.sub(torch.log(sample_freq))

        # return logits and new_labels
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)
        new_targets = Variable(torch.zeros(batch_size).long()).cuda()
        return logits, new_targets

    def full(self, inputs):
        return self.params(inputs)


if __name__ == "__main__":

    ntoken, nhid, ntokens_per_class, bsize, seq_len = 50000, 400, 1, 20, 35
    # PTB 10 per class: 4411000 FULL: 8020000
    # WIKI2 10 per class: 22055000 FULL:40100000
    output = torch.randn((seq_len, bsize, nhid))
    targets = torch.LongTensor(torch.randn(seq_len * bsize).size()).random_(0, ntoken)
    mix_num = None
    test_relaxed = True

    ss = SampledSoftmax(ntoken, nhid, ntokens_per_class)
    out = ss(output.view(-1, output.size(2)), targets)

