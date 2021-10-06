import torch


def log_noise(wv):
    noise = wv + torch.randn(wv.size()) * (torch.log(torch.std(wv)))
    return noise


def normal_noise(wv):
    noise = wv + torch.randn(wv.size()) * torch.std(wv)
    return noise



