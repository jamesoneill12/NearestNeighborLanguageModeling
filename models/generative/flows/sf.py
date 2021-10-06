"""Sequential Flow"""
from torch import nn
from models.layers.ff import FeedForwardNet
import torch
import math


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs = None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples


class SCFLayer(nn.Module):
    def __init__(self, data_dim, n_hidden_layers, n_hidden_units, nonlinearity, transform_function, hidden_order=None,
                 swap_trngen_dirs=False,
                 input_order=None, conditional_inp_dim=None, dropout=[0, 0]):
        super().__init__()

        self.net = FeedForwardNet(data_dim // 2 + conditional_inp_dim, n_hidden_units,
                                  (data_dim - (data_dim // 2)) * transform_function.num_params, n_hidden_layers,
                                  nonlinearity, dropout=dropout[1])

        self.train_func = transform_function.standard if swap_trngen_dirs else transform_function.reverse
        self.gen_func = transform_function.reverse if swap_trngen_dirs else transform_function.standard
        self.input_order = input_order

        self.use_cond_inp = conditional_inp_dim is not None

    def forward(self, inputs):
        """
            Defines the reverse pass which is used during training
            logdet means log det del_y/del_x
        """

        data_dim = len(self.input_order)
        assert data_dim == inputs[0].shape[-1]

        first_indices = torch.arange(len(self.input_order))[
            self.input_order <= data_dim // 2]  # This is <= because input_order goes from 1 to data_dim+1
        second_indices = torch.arange(len(self.input_order))[self.input_order > data_dim // 2]

        if self.use_cond_inp:
            y, logdet, cond_inp = inputs
            net_inp = torch.cat([y[..., first_indices], cond_inp], -1)
        else:
            y, logdet = inputs
            net_inp = y[..., first_indices]

        nn_outp = self.net(net_inp).view(*net_inp.shape[:-1], data_dim - (data_dim // 2),
                                         -1)  # [..., ~data_dim/2, num_params]

        x = torch.tensor(y)
        x[..., second_indices], change_logdet = self.train_func(y[..., second_indices], nn_outp)

        return x, logdet + change_logdet, cond_inp

    def generate(self, inputs):
        """
            Defines the forward pass which is used during testing
            logdet means log det del_y/del_x
        """

        data_dim = len(self.input_order)
        assert data_dim == inputs[0].shape[-1]

        first_indices = torch.arange(len(self.input_order))[
            self.input_order <= data_dim // 2]  # This is <= because input_order goes from 1 to data_dim+1
        second_indices = torch.arange(len(self.input_order))[self.input_order > data_dim // 2]

        if self.use_cond_inp:
            x, logdet, cond_inp = inputs
            net_inp = torch.cat([x[..., first_indices], cond_inp], -1)
        else:
            x, logdet = inputs
            net_inp = x[..., first_indices]

        nn_outp = self.net(net_inp).view(*net_inp.shape[:-1], data_dim - (data_dim // 2),
                                         -1)  # [..., ~data_dim/2, num_params]

        y = torch.tensor(x)
        y[..., second_indices], change_logdet = self.gen_func(x[..., second_indices], nn_outp)

        return y, logdet + change_logdet, cond_inp
