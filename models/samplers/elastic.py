import torch.nn.functional as F
from torch import nn


class ElasticSoftmax(nn.Module):
    def __init__(self, output_size, limit=10.0):
        super(ElasticSoftmax, self).__init__()
        self.lin = nn.Linear(output_size, 1)
        self.limit = limit

    def forward(self, x):
        x = F.softmax(self.lin(x) * self.limit)