import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def order_F_to_C(n):
    idx = np.arange(0, n ** 2)
    idx = idx.reshape(n, n, order="F")
    idx = idx.reshape(n ** 2, order="C")
    idx = list(idx)
    return idx

def init_dct(n, m):
    """ Compute the Overcomplete Discrete Cosinus Transform. """
    oc_dictionary = np.zeros((n, m))
    for k in range(m):
        V = np.cos(np.arange(0, n) * k * np.pi / m)
        if k > 0:
            V = V - np.mean(V)
        oc_dictionary[:, k] = V / np.linalg.norm(V)
    oc_dictionary = np.kron(oc_dictionary, oc_dictionary)
    oc_dictionary = oc_dictionary.dot(np.diag(1 / np.sqrt(np.sum(oc_dictionary ** 2, axis=0))))
    
    idx = np.arange(0, n ** 2)
    idx = idx.reshape(n, n, order="F")
    idx = idx.reshape(n ** 2, order="C")
    oc_dictionary = oc_dictionary[idx, :]
    oc_dictionary = torch.from_numpy(oc_dictionary).float()
    return oc_dictionary


class MLPLam(nn.Module):
    def __init__(self, in_channels, hidden_dims=[128, 64], act=nn.Softplus, norm=nn.LayerNorm):
        super().__init__()
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Linear(in_channels, h_dim, bias=True))
            if norm is not None:
                modules.append(norm(h_dim))
            if act is not None:
                modules.append(act())
            in_channels = h_dim
        modules.append(nn.Linear(hidden_dims[-1], 1, bias=True))
        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        x = self.mlp(x)
        return x


class SC(nn.Module):
    def __init__(self, L, T=5, n_atoms=512, **kwargs):
        super().__init__()
        self.n_atoms = n_atoms
        self.L =  L
        self.T = T

        self.soft_comp = nn.Parameter(torch.zeros(n_atoms))
        self.Identity = nn.Parameter(torch.eye(n_atoms))

        # disable gradient 
        self.soft_comp.requires_grad = False
        self.Identity.requires_grad = False

        self.Dict = nn.Parameter(torch.randn(L, n_atoms)) # we found random intialization is even better
        self.c = nn.Parameter(torch.linalg.norm(self.Dict, ord=2) ** 2)
    
        # Lambda Layer
        self.lam_layer = MLPLam(L, **kwargs)

        self._init_weights()
    
    def soft_thresh(self, x, l):
        return torch.sign(x) * torch.max(torch.abs(x) - l, self.soft_comp)

    def _init_weights(self):
        with torch.no_grad():
            init.kaiming_uniform_(self.Dict)
            self.Dict.data = F.normalize(self.Dict.data, p=2, dim=0) 
            self.c.data = torch.linalg.norm(self.Dict.data, ord=2) ** 2
    
    def forward(self, x):
        lam = self.lam_layer(x)    
        
        # <------------- Sparse DicL ------------->
        l = lam / self.c # compute lambda
        y = torch.matmul(x, self.Dict) # project y

        ################ ISTA update ################
        S = self.Identity - (1 / self.c) * self.Dict.t().mm(self.Dict) 
        S = S.t()

        z = self.soft_thresh(y, l)
        for _ in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        return z