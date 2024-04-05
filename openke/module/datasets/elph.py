import os
from time import time

import torch
from torch_geometric.data import Dataset
import scipy.sparse as ssp

from openke.module.datasets.utils import get_src_dst_degree
from openke.module.hashes.hashing import ElphHashes


class HashDataset(Dataset):
    """
    A dataset class for MinHash & Hyperloglog features
    """

    def __init__(
            self, edge_index, num_nodes, max_hash_hops=3, hll_p=8,  minhash_num_perm=128, **kwargs):
        self.max_hash_hops = max_hash_hops
        self.hll_p = hll_p
        self.elph_hashes = ElphHashes(max_hash_hops=self.max_hash_hops, hll_p=hll_p,  minhash_num_perm=minhash_num_perm)  # object for hash and subgraph feature operations
        self.subgraph_features = None
        self.hashes = None
        self.device = edge_index.device
        super(HashDataset, self).__init__(None)

        self.links = self.edge_index = edge_index
        self.labels = None
        self.edge_weight = torch.ones(self.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (self.edge_weight, (self.edge_index[0].to('cpu'), self.edge_index[1].to('cpu'))),
            shape=(num_nodes, num_nodes)
        )
        self.edge_index[0].to(self.device)
        self.edge_index[1].to(self.device)
        self.edge_index.to(self.device)

        self.degrees = torch.tensor(self.A.sum(axis=0, dtype=float), dtype=torch.float).flatten()
        # pre-process subgraph features
        self._preprocess_hash_features(num_nodes)

    def _preprocess_hash_features(self, num_nodes):
        print('generating Minhash & Hyperloglog')
        start_time = time()
        self.hashes, self.cards = self.elph_hashes.build_hash_tables(num_nodes, self.edge_index)
        print("Preprocessed hashes in: {:.2f} seconds".format(time() - start_time))

    def len(self):
        return len(self.links)

    def get_hashes(self):
        return self.hashes
    
    def get_cards(self):
        return self.cards

    def get(self, idx):
        return None