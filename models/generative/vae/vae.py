from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


""" --------------------- Variational AE ------------------------- """

class VAE(nn.Module):
    def __init__(self, x_shape, hidden_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(x_shape, 300)
        self.fc21 = nn.Linear(300, hidden_dim)
        self.fc22 = nn.Linear(300, hidden_dim)
        self.fc3 = nn.Linear(20, 300)
        self.fc4 = nn.Linear(300, x_shape)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[0]))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

