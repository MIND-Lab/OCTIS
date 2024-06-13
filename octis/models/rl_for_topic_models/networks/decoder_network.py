"""PyTorch class for feed-forward decoder network."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
import numpy as np

from octis.models.rl_for_topic_models.networks.inference_network \
    import InferenceNetwork


class DecoderNetwork(nn.Module):

    """RLTM Network."""

    def __init__(
            self, input_size, bert_size, n_components=10,
            hidden_sizes=(128, 128), activation='gelu',
            inference_dropout=0.2, policy_dropout=0.0, kl_multiplier=1.0):
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            bert_size : int, dimension of BERT input
            n_components : int, number of topic components, (default 10)
            hidden_sizes : tuple, length = n_layers, (default (128, 128))
            activation : string, default 'gelu'
            inference_dropout : float, inference dropout to use (default 0.2)
            policy_dropout : float, policy dropout to use (default 0.0)
            kl_multiplier : float or int, multiplier on the KL divergence
                (default 1.0)
        """
        super(DecoderNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(bert_size, int), "input_size must by type int."
        assert (isinstance(n_components, int) or
                isinstance(n_components, np.int64)) \
            and n_components > 0, "n_components must be type int > 0."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'sigmoid', 'tanh',
                              'leakyrelu', 'rrelu', 'elu', 'selu', 'gelu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'tanh'," \
            " 'leakyrelu', 'rrelu', 'elu', 'selu', or 'gelu'."
        assert inference_dropout >= 0, "inference dropout must be >= 0."
        assert policy_dropout >= 0, "policy dropout must be >= 0."
        assert isinstance(kl_multiplier, float) \
            or isinstance(kl_multiplier, int), \
            "kl_multiplier must be a float or int"

        self.n_components = n_components
        self.kl_multiplier = float(kl_multiplier)

        self.mu_inference = InferenceNetwork(
            bert_size, n_components, hidden_sizes,
            activation=activation, dropout=inference_dropout)
        self.log_sigma_inference = InferenceNetwork(
            bert_size, n_components, hidden_sizes,
            activation=activation, dropout=inference_dropout)

        if torch.cuda.is_available():
            self.mu_inference = self.mu_inference.cuda()
            self.log_sigma_inference = self.log_sigma_inference.cuda()

        self.prior_mean = torch.Tensor(torch.zeros(n_components))
        if torch.cuda.is_available():
            self.prior_mean = nn.Parameter(self.prior_mean.cuda())
        self.prior_mean = nn.Parameter(self.prior_mean)

        self.prior_variance = torch.Tensor(torch.ones(n_components))
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta = 0.02 * torch.randn((n_components, input_size))
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)

        self.posterior_log_sigma_norm = nn.LayerNorm(
            n_components, elementwise_affine=False)
        self.dropout = nn.Dropout(p=policy_dropout)
        self.beta_norm = nn.LayerNorm(input_size, elementwise_affine=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def kl_divergence(self, p_mean, p_variance, q_mean, q_variance):
        var_division = torch.sum(p_variance ** 2 / q_variance ** 2, dim=-1)
        diff_term = torch.sum((q_mean - p_mean) ** 2 / q_variance ** 2, dim=-1)
        logvar_det_division = torch.sum(
            torch.log(q_variance ** 2) - torch.log(p_variance ** 2), dim=-1)
        return 0.5 * (var_division + diff_term
                      - self.n_components + logvar_det_division)

    def loss_fn(self, bow, word_dist, posterior_mu,
                posterior_log_sigma, epsilon=1e-8):
        # forward KL divergence
        unscaled_kl = self.kl_divergence(
            posterior_mu, torch.exp(posterior_log_sigma),
            self.prior_mean, self.prior_variance)

        kl = self.kl_multiplier * unscaled_kl

        # reconstruction loss (log likelihood)
        nll = -1.0 * torch.sum(bow * torch.log(word_dist + epsilon), dim=-1)

        reward = nll + kl
        return reward.mean()

    def forward(self, x_bow, x_bert):
        """Forward pass."""
        # inference networks
        posterior_mu = self.mu_inference(x_bert)
        posterior_log_sigma_unnormalized = self.log_sigma_inference(x_bert)
        posterior_log_sigma = self.posterior_log_sigma_norm(
            posterior_log_sigma_unnormalized)
        posterior_distribution = Normal(
            posterior_mu, torch.exp(posterior_log_sigma))

        # RL policy
        action = posterior_distribution.rsample()
        self.action = action
        policy = (1 / (torch.exp(posterior_log_sigma)
                       * math.sqrt(2 * math.pi))) \
            * torch.exp(-1.0 * (action - posterior_mu) ** 2
                        / (2 * torch.exp(posterior_log_sigma) ** 2))
        policy = self.dropout(policy)

        # product of experts
        word_dist = F.softmax(
            self.beta_norm(torch.matmul(policy, self.beta)), dim=-1)

        # loss
        loss = self.loss_fn(
            x_bow, word_dist, posterior_mu, posterior_log_sigma)
        return loss

    def get_topic_distribution(self, x_bert):
        with torch.no_grad():
            # inference networks
            posterior_mu = self.mu_inference(x_bert)
            posterior_log_sigma_unnormalized = self.log_sigma_inference(x_bert)
            posterior_log_sigma = self.posterior_log_sigma_norm(
                posterior_log_sigma_unnormalized)
            posterior_distribution = Normal(
                posterior_mu, torch.exp(posterior_log_sigma))

            action = posterior_distribution.rsample()
            softmax_action = F.softmax(action, dim=-1)
            return softmax_action
