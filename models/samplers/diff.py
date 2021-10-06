"""
Differentiable Scheduled Sampling
"""

import torch
import torch.nn.functional as F
from torch import nn


class DifferentiableSchedule(nn.Module):
    """Learns a schedule sampling rate,
    if num_insts pass we have a rate for each instance"""
    def __init__(self, num_instances=None, sent_len=None):
        super(DifferentiableSchedule, self).__init__()

        self.num_instances = num_instances
        self.batch_len = sent_len

        """Global sampling rate"""
        if num_instances is not None and sent_len is None:
            self.fc = nn.Linear(1)
            """Sampling rate per instance"""
        elif num_instances is None and sent_len is not None:
            self.fc = nn.Linear(sent_len)
            """Sampling rate per instance"""
        elif num_instances is not None and sent_len is None:
            self.fc = nn.Linear(num_instances)
            """Sampling rate per time step and instance"""
        else:
            self.fc = nn.Linear(num_instances, sent_len)

    """Id x_inds passed then we update the sampling rate"""
    def forward(self, x, k=None):
        x = self.fc(x)
        x = F.sigmoid(x)
        if k is not None:
            x, inds = torch.topk(x, k)

        return x