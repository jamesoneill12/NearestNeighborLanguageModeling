import torch
from torch import nn
import torch.nn.functional as F

""" --------------------------- Adversarial AE ---------------------------------- """

# Encoder
class Q_net(nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)

    def forward(self, x):
        x = F.droppout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.droppout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss


# Decoder
class P_net(nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)


# Discriminator
class D_net_gauss(nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))

