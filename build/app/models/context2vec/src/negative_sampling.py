from torch import tensor
import torch
import torch.nn as nn
import numpy as np

# from models.context2vec.src.walker_alias import WalkerAlias
# from models.context2vec.src.utils import init_embeddings

class NegativeSampling(nn.Module):
    def __init__(self, embed_size, counter, num_neg, power, device, pad_idx):
        super().__init__()
        self.counter = counter
        self.num_neg = num_neg
        self.power = power
        self.device = device
        
        self.W = nn.Embedding(len(counter), embedding_dim=embed_size, padding_idx=pad_idx)
        init_embeddings(self.W)
        # self.W.weight.data.zero_()
        self.log_loss = nn.LogSigmoid()
        self.sampler = WalkerAlias(np.power(counter, power))
        
    def negative_sampling(self, shape):
        return tensor(self.sampler.sample(shape=shape), dtype=torch.long, device=self.device)
    
    def forward(self, X, context):
        batch_size, seq_len = X.size()
        embedding = self.W(X)
        pos_loss = self.log_loss((embedding * context).sum(2))

        neg_samples = self.negative_sampling(shape=(batch_size, seq_len, self.num_neg))
        neg_embedding = self.W(neg_samples)
        neg_loss = self.log_loss((-neg_embedding * context.unsqueeze(2)).sum(3)).sum(2)
        return -(pos_loss + neg_loss).sum()
        
        
        