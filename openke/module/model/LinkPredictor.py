
from time import time
import logging
from collections import OrderedDict

import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from .Model import Model

# hash_hops * (hash_hops ** 2 + 3) * 2
# simple MLP as a Link Predictor
class LinkPredictor(Model):
    def __init__(self, ent_tot, rel_tot, label_dropout=0.5, max_hash_hops=3, hidden_dim=128):
        super(LinkPredictor, self).__init__(ent_tot, rel_tot)
        self.label_dropout = label_dropout
        self.dim = max_hash_hops * (max_hash_hops ** 2 + 3)
        self.hidden_dim = hidden_dim
        self.sf_lin_layer = nn.Sequential(OrderedDict([
            ('dense1', Linear(self.dim, self.hidden_dim)),
            ('norm1', torch.nn.BatchNorm1d(self.hidden_dim)),
            ('relu1', nn.ReLU()),
            ('dense2', Linear(self.hidden_dim, self.hidden_dim)),
            ('norm2', torch.nn.BatchNorm1d(self.hidden_dim)),
            ('relu2', nn.ReLU()),
            ('dense3', Linear(self.hidden_dim, self.hidden_dim)),
            ('norm3', torch.nn.BatchNorm1d(self.hidden_dim)),
            ('relu3', nn.ReLU()),
        ]))
        out_channels = self.hidden_dim
        self.lin = Linear(out_channels, 1)

    # sf = subgraph features
    def forward(self, data):
        #print(data)
        sf = data["subgraph_feature"]
        x = self.sf_lin_layer(sf)
        x = self.lin(x)
        x = F.sigmoid(x)
        return x

    def print_params(self):
        print(f'model bias: {self.lin.bias.item():.3f}')
        print('model weights')
        for idx, elem in enumerate(self.lin.weight.squeeze()):
            if idx < self.dim:
                print(f'{self.idx_to_dst[idx % self.emb_dim]}: {elem.item():.3f}')
            else:
                print(f'feature {idx - self.dim}: {elem.item():.3f}')

    def predict(self, data):
        #pdb.set_trace()
        self.training = False
        score = self.forward(data)
        return score.cpu().data.numpy()
