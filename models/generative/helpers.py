import torch
from torch import nn


def dim_fixer(word_dims, fixed_dim):
    if word_dims < 101:
        if fixed_dim is not None:
            hidden_dim = fixed_dim
        else:
            fixed_dim = 50
    else:
        if fixed_dim is not None:
            hidden_dim = fixed_dim
        else:
            fixed_dim = 100
    return hidden_dim


def to_sparse(x, cuda=True):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    if cuda:
        sparse_tensortype = getattr(torch.cuda.sparse, x_typename)
    else:
        sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def choose_activations(activations):
    if activations == 'sigmoid':
        return nn.Sigmoid()
    elif activations == 'relu':
        return nn.ReLU()
    elif activations == 'leaky':
        return nn.LeakyReLU()
    else:
        return nn.Tanh()