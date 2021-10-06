import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn


class GumbellSoftmax(nn.Module):
    def __init__(self):
        super(GumbellSoftmax, self).__init__()
        self.Lambda = 1

    def forward(self, x, lb=1e-10):
        x = torch.log(x) - torch.log(-torch.log(torch.rand(x.size())+lb))
        x = F.softmax(x/self.Lambda)
        return x


def _sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-20):
    """
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = logits.dim()
    gumbel_noise = Variable(_sample_gumbel(logits.size(), eps=eps, out=logits.data.new()))
    y = logits + gumbel_noise
    return F.softmax(y / tau, dims - 1)


def gumbel_softmax(logits, tau=0.8, hard=False, eps=1e-10):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: `[batch_size, n_class]` unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if ``True``, take `argmax`, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = _gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        _, k = y_soft.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros_like(logits).scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y


# newer and simpler, taken from https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530
"""
def sample_gumbel(input):
    noise = torch.rand(input.size())
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise).cuda()


def gumbel_softmax_sample(input, temperature = 1):
    noise = sample_gumbel(input)
    x = (input + noise) / temperature
    x = F.log_softmax(x)
    return x.view_as(input)
"""




if __name__ == "__main__":
    logits = F.softmax(Variable(torch.randn(10, 10)))
    print(logits)
    y_draw = gumbel_softmax(logits, hard=False)
    print(y_draw)


