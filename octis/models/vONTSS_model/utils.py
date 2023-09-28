import torch

def kld_normal(mu, log_sigma):
    """KL divergence to standard normal distribution.
    mu: batch_size x dim
    log_sigma: batch_size x dim
    """
    #normal distribution KL divergence of two gaussian
    #https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)


