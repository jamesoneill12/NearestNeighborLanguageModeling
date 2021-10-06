from torch.autograd import Variable
import torch
from torch import nn
import torch.nn.functional as F
from .nearest_embed import NearestEmbed

""" -------------------------- VQ-VAE ------------------------------- """

class VQ_VAE(nn.Module):
    """Vector Quantized AutoEncoder for mnist"""
    def __init__(self, in_dim=784, hidden_dims=[200, 400], k=10, vq_coef=0.2, comit_coef=0.4, **kwargs):
        super(VQ_VAE, self).__init__()

        self.emb_size = k

        self.layers = []
        for i in range(2):
            for j in range(1, len(hidden_dims)):
                dims = (in_dim, dims[j]) if j == 0 else dims(dims[j-1], dims[j])
                if i == 2: dims = reversed(dims)
                self.layers.append(nn.Linear(dims))
        self.layers = nn.ModuleList(self.layers)

        self.emb = NearestEmbed(k, self.emb_size)
        self.hidden = hidden_dims[-1] if type(hidden_dims)==list else hidden_dims
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.ce_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.fc2(h1)
        return h2.view(-1, self.emb_size, int(self.hidden / self.emb_size))

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.tanh(self.fc4(h3))

    def forward(self, x):
        z_e = self.encode(x.view(-1, 784))
        z_q, _ = self.emb(z_e, weight_sg=True).view(-1, self.hidden)
        emb, _ = self.emb(z_e.detach()).view(-1, self.hidden)
        return self.decode(z_q), z_e, emb

    def sample(self, size):
        sample = Variable(torch.randn(size, self.emb_size, int(self.hidden / self.emb_size)))
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        sample = self.decode(emb(sample).view(-1, self.hidden)).cpu()
        return sample

    def loss_function(self, x, recon_x, z_e, emb):
        self.ce_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784))
        self.vq_loss = F.mse_loss(emb, z_e.detach())
        self.commit_loss = F.mse_loss(z_e, emb.detach())

        return self.ce_loss + self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}