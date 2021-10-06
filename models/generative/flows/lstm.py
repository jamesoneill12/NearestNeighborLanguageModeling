import math
import torch
from torch import nn
from models.networks.generative.flows.affine import Affine
from models.networks.generative.flows.NLSq import NLSq
from models.layers.lstm_af import LSTM_AFLayer


# Full flow combining multiple layers
class LSTMFlow(nn.Module):
    def __init__(self, inp_dim, n_hidden_layers, n_hidden_units, dropout_p, num_flow_layers, transform_function,
                 rnn_cond_dim=None, swap_trngen_dirs=False,
                 sequential_training=False, reverse_ordering=False, hiddenflow_params={},
                 dlocs=[], notimecontext=False):
        super().__init__()

        if transform_function == 'affine':
            transform_function = Affine
        elif transform_function == 'nlsq':
            transform_function = NLSq
        else:
            raise NotImplementedError('Only the affine and nlsq transformation functions have been implemented')

        # Note: This ordering is the ordering as applied during training
        flow_layers = []
        reverse_inps = False

        # This is neccessary so that q(z) and p(z) are based on the same ordering if there are
        # an even number of layers and IAF posterior is used
        if swap_trngen_dirs and num_flow_layers % 2 == 0:
            reverse_inps = True

        # This is needed after the previous line, because if using sequential training for p (i.e. IAF prior) you
        # don't want to start with reversed inputs if you have an even number of flow layers
        if sequential_training:
            swap_trngen_dirs = not swap_trngen_dirs

        for i in range(num_flow_layers):
            flow_layers.append(LSTM_AFLayer(i, inp_dim, n_hidden_layers, n_hidden_units, dropout_p, transform_function,
                                            rnn_cond_dim=rnn_cond_dim, swap_trngen_dirs=swap_trngen_dirs,
                                            reverse_inps=reverse_inps,
                                            hiddenflow_params=hiddenflow_params, dlocs=dlocs,
                                            notimecontext=notimecontext))
            if reverse_ordering:
                reverse_inps = not reverse_inps

        self.flow = nn.Sequential(*flow_layers)
        self.use_rnn_cond_inp = rnn_cond_dim is not None
        self.sequential_training = sequential_training

    def forward(self, y, hiddens, lengths, rnn_cond_inp=None):
        """
            Defines the reverse pass which is used during training
            logdet means log det del_y/del_x
        """
        # if self.use_cond_inp:
        #    y, hiddens, cond_inp = inputs
        # else:
        #    y, hiddens = inputs

        if self.use_rnn_cond_inp and rnn_cond_inp is None:
            raise ValueError("use_rnn_cond_inp is set but rnn_cond_inp is None in forward")

        logdet = torch.zeros(y.shape[:-1], device=y.device)

        if self.sequential_training:
            x = y
            for flow_layer in reversed(self.flow):
                x, logdet, hiddens, _, _ = flow_layer.generate([x, logdet, hiddens, rnn_cond_inp, lengths])
        else:
            x, logdet, hiddens, _, _ = self.flow([y, logdet, hiddens, rnn_cond_inp, lengths])

        return x, logdet, hiddens

    def generate(self, x, hiddens, lengths, rnn_cond_inp=None):
        """
            Defines the forward pass which is used during testing
            logdet means log det del_y/del_x
        """

        if self.use_rnn_cond_inp and rnn_cond_inp is None:
            raise ValueError("use_rnn_cond_inp is set but rnn_cond_inp is None in generate")

        logdet = torch.zeros(x.shape[:-1], device=x.device)

        if self.sequential_training:
            y, logdet, hiddens, _, _ = self.flow([x, logdet, hiddens, rnn_cond_inp, lengths])
        else:
            y = x
            for flow_layer in reversed(self.flow):
                y, logdet, hiddens, _, _ = flow_layer.generate([y, logdet, hiddens, rnn_cond_inp, lengths])

        return y, logdet, hiddens

    def init_hidden(self, batch_size):
        return [fl.init_hidden(batch_size) for fl in self.flow]


# Prior using the LSTMFlow

class AFPrior(nn.Module):
    def __init__(self, hidden_size, zsize, dropout_p, dropout_locations, prior_type, num_flow_layers, rnn_layers,
                 max_T=-1,
                 transform_function='affine', hiddenflow_params={}):
        super().__init__()

        sequential_training = prior_type == 'IAF'
        notimecontext = prior_type == 'hiddenflow_only'

        dlocs = []
        if 'prior_rnn' in dropout_locations:
            dlocs.append('recurrent')
            dlocs.append('rnn_outp')
        if 'prior_rnn_inp' in dropout_locations:
            dlocs.append('rnn_inp')
        if 'prior_ff' in dropout_locations:
            dlocs.append('ff')

        self.flow = LSTMFlow(zsize, rnn_layers, hidden_size, dropout_p, num_flow_layers,
                             transform_function, rnn_cond_dim=2 * max_T,
                             sequential_training=sequential_training, hiddenflow_params=hiddenflow_params, dlocs=dlocs,
                             notimecontext=notimecontext)

        self.dropout = nn.Dropout(dropout_p)

        self.hidden_size = hidden_size
        self.zsize = zsize
        self.dropout_locations = dropout_locations

    def evaluate(self, z, lengths_s, cond_inp_s=None):
        """
            z is [T, B, s, E]
            output is log_p_z [T, B, s]
        """
        T, B, ELBO_samples = z.shape[:3]

        hidden = self.flow.init_hidden(B)
        hidden = [tuple(
            h[:, :, None, :].repeat(1, 1, ELBO_samples, 1).view(-1, ELBO_samples * B, self.hidden_size) for h in
            hidden_pl) for hidden_pl in hidden]

        if 'z_before_prior' in self.dropout_locations:
            z = self.dropout(z)

        z = z.view(T, B * ELBO_samples, z.shape[-1])
        eps, logdet, _ = self.flow(z, hidden, lengths_s, rnn_cond_inp=cond_inp_s)
        eps = eps.view(T, B, ELBO_samples, self.zsize)
        logdet = logdet.view(T, B, ELBO_samples)

        log_p_eps = -1 / 2 * (math.log(2 * math.pi) + eps.pow(2)).sum(-1)  # [T, B, s]
        log_p_z = log_p_eps - logdet

        return log_p_z

    def generate(self, lengths, cond_inp=None, temp=1.0):
        T = torch.max(lengths)
        B = lengths.shape[0]

        hidden = self.flow.init_hidden(B)

        eps = torch.randn((T, B, self.zsize), device=hidden[0][0].device) * temp
        z, logdet, _ = self.flow.generate(eps, hidden, lengths, rnn_cond_inp=cond_inp)

        log_p_eps = -1 / 2 * (math.log(2 * math.pi) + eps.pow(2)).sum(-1)  # [T, B]
        log_p_zs = log_p_eps - logdet

        return z, log_p_zs