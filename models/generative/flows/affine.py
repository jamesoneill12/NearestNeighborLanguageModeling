import torch


class Affine():
    num_params = 2

    @staticmethod
    def get_pseudo_params(nn_outp):
        a = nn_outp[..., 0]  # [B, D]
        var_outp = nn_outp[..., 1]

        b = torch.exp(0.5 * var_outp)
        logbsq = var_outp

        return a, logbsq, b

    @staticmethod
    def standard(x, nn_outp):
        a, logbsq, b = Affine.get_pseudo_params(nn_outp)

        y = a + b * x
        logdet = 0.5 * logbsq.sum(-1)

        return y, logdet

    @staticmethod
    def reverse(y, nn_outp):
        a, logbsq, b = Affine.get_pseudo_params(nn_outp)

        x = (y - a) / b
        logdet = 0.5 * logbsq.sum(-1)

        return x, logdet