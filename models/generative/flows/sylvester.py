import torch
from torch import nn
from torch.autograd import Variable

class Sylvester(nn.Module):
    """
    Sylvester normalizing flow.
    """

    def __init__(self, num_ortho_vecs):

        super(Sylvester, self).__init__()

        self.num_ortho_vecs = num_ortho_vecs

        self.h = nn.Tanh()

        triu_mask = torch.triu(torch.ones(num_ortho_vecs, num_ortho_vecs), diagonal=1).unsqueeze(0)
        diag_idx = torch.arange(0, num_ortho_vecs).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        """
        All flow parameters are amortized. Conditions on diagonals of R1 and R2 for invertibility need to be satisfied
        outside of this function. Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param q_ortho: shape (batch_size, z_size , num_ortho_vecs)
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """

        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        r1_hat = r1
        r2_hat = r2

        qr2 = torch.bmm(q_ortho, r2_hat.transpose(2, 1))
        qr1 = torch.bmm(q_ortho, r1_hat)

        r2qzb = torch.bmm(zk, qr2) + b
        z = torch.bmm(self.h(r2qzb), qr1.transpose(2, 1)) + zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class TriangularSylvester(nn.Module):
    """
    Sylvester normalizing flow with Q=P or Q=I.
    """

    def __init__(self, z_size):

        super(TriangularSylvester, self).__init__()

        self.z_size = z_size
        self.h = nn.Tanh()

        diag_idx = torch.arange(0, z_size).long()
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, b, permute_z=None, sum_ldj=True):
        """
        All flow parameters are amortized. conditions on diagonals of R1 and R2 need to be satisfied
        outside of this function.
        Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        with Q = P a permutation matrix (equal to identity matrix if permute_z=None)
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """

        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        if permute_z is not None:
            # permute order of z
            z_per = zk[:, :, permute_z]
        else:
            z_per = zk

        r2qzb = torch.bmm(z_per, r2.transpose(2, 1)) + b
        z = torch.bmm(self.h(r2qzb), r1.transpose(2, 1))

        if permute_z is not None:
            # permute order of z again back again
            z = z[:, :, permute_z]

        z += zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)