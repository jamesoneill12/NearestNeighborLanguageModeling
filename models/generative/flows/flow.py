"""Flow for Discrete Sequence Normalization Flow"""
# Full flow combining multiple layers
from torch import nn
import torch
from models.layers.aflayer import AFLayer
from models.layers.scflayer import SCFLayer
from models.networks.generative.flows.affine import Affine
from models.networks.generative.flows.NLSq import NLSq


class Flow(nn.Module):
    def __init__(self, data_dim, n_hidden_layers, n_hidden_units, nonlinearity, num_flow_layers, transform_function,
                 iaf_like=False, hidden_order='sequential',
                 swap_trngen_dirs=False, conditional_inp_dim=None, dropout=[0, 0], reverse_between_layers=True,
                 scf_layers=False, reverse_first_layer=False):
        super().__init__()

        if transform_function == 'affine':
            transform_function = Affine
        elif transform_function == 'nlsq':
            transform_function = NLSq
        elif transform_function != Affine and transform_function != NLSq:  # Can pass string or actual class
            raise NotImplementedError('Only the affine transformation function has been implemented')

        if scf_layers:
            AutoregressiveLayer = SCFLayer
        else:
            AutoregressiveLayer = AFLayer

        # Note: This ordering is the ordering as applied to go from data -> base
        flow_layers = []

        input_order = torch.arange(data_dim) + 1

        if reverse_first_layer:
            input_order = reversed(input_order)

        for i in range(num_flow_layers):
            flow_layers.append(
                AutoregressiveLayer(data_dim, n_hidden_layers, n_hidden_units, nonlinearity, transform_function,
                                    hidden_order=hidden_order, swap_trngen_dirs=swap_trngen_dirs,
                                    input_order=input_order,
                                    conditional_inp_dim=conditional_inp_dim, dropout=dropout))
            if reverse_between_layers:
                input_order = reversed(input_order)

        self.flow = nn.Sequential(*flow_layers)
        self.use_cond_inp = conditional_inp_dim is not None

    def forward(self, inputs):
        """
            Defines the reverse pass which is used during training
            logdet means log det del_y/del_x
        """
        if self.use_cond_inp:
            y, cond_inp = inputs
        else:
            y = inputs

        logdet = torch.zeros(y.shape[:-1], device=y.device)

        if self.use_cond_inp:
            x, logdet, _ = self.flow([y, logdet, cond_inp])
        else:
            x, logdet = self.flow([y, logdet])

        return x, logdet

    def generate(self, inputs):
        """
            Defines the forward pass which is used during testing
            logdet means log det del_y/del_x
        """

        if self.use_cond_inp:
            x, cond_inp = inputs
        else:
            x = inputs

        logdet = torch.zeros(x.shape[:-1], device=x.device)
        y = x
        for flow_layer in reversed(self.flow):
            if self.use_cond_inp:
                y, logdet, _ = flow_layer.generate([y, logdet, cond_inp])
            else:
                y, logdet = flow_layer.generate([y, logdet])

        return y, logdet