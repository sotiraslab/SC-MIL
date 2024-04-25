import torch.nn as nn
from models.utils import *


class NonparamMIL(nn.Module):
    def __init__(self, in_features, dimR=True, m_dim=512, mode='mean', num_cls=2, droprate=0):
        super().__init__()
        self.mode = mode
        if dimR:
            self.dimReduction = DimReduction(in_features, m_dim)
            self.classifier = Classifier_1fc(m_dim, num_cls, droprate)
        else:
            self.dimReduction = nn.Identity()
            self.classifier = Classifier_1fc(in_features, num_cls, droprate)

    def forward(self, x):
        x = self.dimReduction(x)
        if self.mode == 'max':
            x = torch.amax(x, dim=0, keepdim=True)  # KxL
        elif self.mode == 'mean':
            x = torch.mean(x, dim=0, keepdim=True)  # KxL
        logit = self.classifier(x)

        return logit