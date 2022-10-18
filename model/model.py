import torch
import torch.nn as nn
from lib.ssn.ssn import ssn_iter, sparse_ssn_iter


class SSNModel(nn.module):
    def __init__(self, n_spix, training, n_iter=10):
        super().__init__()
        self.n_spix = n_spix
        self.n_iter = n_iter
        self.training = training

    def forward(self, x):

        if self.training:
            return ssn_iter(x, self.n_spix, self.n_iter)
        else:
            sparse_ssn_iter(x, self.n_spix, self.n_iter)
