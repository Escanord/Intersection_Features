import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .Loss import Loss


class NLLLoss(Loss):
    def __init__(self, adv_temperature=None):
        super(NLLLoss, self).__init__()
        self.DEFAULT_CLAM_EXP_LOWER = -75.0
        self.DEFAULT_CLAM_EXP_UPPER = 75.0

    def get_weights(self, n_score):
        return F.softmax(-n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        p_score = torch.clamp(
            p_score, self.DEFAULT_CLAM_EXP_LOWER, self.DEFAULT_CLAM_EXP_UPPER
        )
        n_score = torch.clamp(
            n_score, self.DEFAULT_CLAM_EXP_LOWER, self.DEFAULT_CLAM_EXP_UPPER
        )

        p_exp = torch.exp(p_score).squeeze()
        n_exp = torch.exp(n_score)

        softmax_score = 1 - p_exp / (torch.sum(n_exp, 1) + p_exp)
        #pdb.set_trace()

        return -torch.log(softmax_score.mean())

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()
