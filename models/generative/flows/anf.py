"""Activation Normalization Flow"""
from torch import nn
import torch


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.initialized == False:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        if mode == 'direct':
            return (
                inputs - self.bias) * torch.exp(self.weight), self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs * torch.exp(
                -self.weight) + self.bias, -self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)
