import torch
import torch.nn as nn

from models.utils import *
    

class ABMIL(nn.Module):
    def __init__(self, in_features, m_dim=None, attn_mode='gated', D=128, K=1, num_cls=2, droprate=0, sc=None):
        super().__init__()
        if m_dim is not None:
            self.dimReduction = DimReduction(in_features, m_dim)
            self.m_dim = m_dim
        else:
            self.dimReduction = nn.Identity()
            self.m_dim = in_features
        
        if sc is None:
            self.sc = nn.Identity()
        else:
            self.sc = sc
            self.m_dim = self.sc.n_atoms
            
        if attn_mode =='gated':
            self.attn = Attention_Gated(self.m_dim, D, K)  
        else:
            self.attn = Attention(self.m_dim, D, K)  

        self.classifier = Classifier_1fc(self.m_dim*K, num_cls, droprate)
        
    def forward(self, x):
        x = self.dimReduction(x)
        x = self.sc(x)
        AA = self.attn(x)
        afeat = torch.mm(AA, x)
        logit = self.classifier(afeat)
        return logit