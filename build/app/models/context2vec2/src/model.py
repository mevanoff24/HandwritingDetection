import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal
from torch import tensor
import numpy as np
import math

from src.negative_sampling import NegativeSampling
from src.utils import init_embeddings


def create_embedding_layer(vocab_size, word_embed_size, pad_idx):
    return nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=word_embed_size,
                                    padding_idx=pad_idx)

def create_rnn_layer(word_embed_size, hidden_size, n_layers, batch_first, layer_type=nn.LSTM):
    return layer_type(input_size=word_embed_size,
                               hidden_size=hidden_size,
                               num_layers=n_layers,
                               batch_first=batch_first) 

class Context2vec(nn.Module):
    def __init__(self, vocab_size, counter, word_embed_size, hidden_size, n_layers, bidirectional, dropout,
                 pad_idx, device, inference):

        super().__init__()
        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.inference = inference
        self.rnn_output_size = hidden_size
        
        # embedding
        self.left2right_embed = create_embedding_layer(vocab_size, word_embed_size, pad_idx)
        self.right2left_embed = create_embedding_layer(vocab_size, word_embed_size, pad_idx)
        for embed in [self.left2right_embed, self.right2left_embed]:
            init_embeddings(embed)
        # rnn
        self.left2right_rnn = create_rnn_layer(word_embed_size, hidden_size, n_layers, batch_first=True)
        self.right2left_rnn = create_rnn_layer(word_embed_size, hidden_size, n_layers, batch_first=True)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # loss
        self.neg_sample_loss = NegativeSampling(hidden_size, counter, pad_idx=pad_idx, num_neg=10, power=0.75,
                                          device=device) # num_neg=10, power=0.75 used in paper
        
        self.top_model = NeuralNet(input_size=hidden_size*2, mid_size=hidden_size*2, output_size=hidden_size,
                                                               dropout=dropout)
        
    def forward(self, X, y, target_pos=None):
        batch_size, seq_len = X.size()
        X_reversed = X.flip(1)[:, :-1]
        X = X[:, :-1]
        
        left2right_embed = self.left2right_embed(X)
        right2left_embed = self.right2left_embed(X_reversed)
        
        left2right_out, _ = self.left2right_rnn(left2right_embed)
        right2left_out, _ = self.right2left_rnn(right2left_embed)
        
        left2right_out = left2right_out[:, :-1, :]
        right2left_out = right2left_out[:, :-1, :].flip(1)
        # TESTING
        if self.inference:
            left2right_out = left2right_out[0, target_pos]
            right2left_out = right2left_out[0, target_pos]
            out = self.top_model(torch.cat((left2right_out, right2left_out), dim=0))
            return out
        # TRAINING 
        else:
            out = self.top_model(torch.cat((left2right_out, right2left_out), dim=2)) # dim = 2
            loss = self.neg_sample_loss(y, out)
            return loss 
        
    def run_inference(self, input_tokens, target, target_pos, k=10):
        context_vector = self.forward(input_tokens, target=None, target_pos=target_pos)
        if target is None:
            topv, topi = ((self.neg_sample_loss.W.weight*context_vector).sum(dim=1)).data.topk(k)
            return topv, topi
        else:
            context_vector /= torch.norm(context_vector, p=2)
            target_vector = self.neg_sample_loss.W.weight[target]
            target_vector /= torch.norm(target_vector, p=2)
            similarity = (target_vector * context_vector).sum()
            return similarity.item()
        
        
        
class NeuralNet(nn.Module):

    def __init__(self, input_size, mid_size, output_size, n_layers=2, dropout=0.3, activation_function='relu'):
        super().__init__()
        self.input_size = input_size
        self.mid_size = mid_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop = nn.Dropout(dropout)

        self.MLP = nn.ModuleList()
        if n_layers == 1:
            self.MLP.append(nn.Linear(input_size, output_size))
        else:
            self.MLP.append(nn.Linear(input_size, mid_size))
            for _ in range(n_layers - 2):
                self.MLP.append(nn.Linear(mid_size, mid_size))
            self.MLP.append(nn.Linear(mid_size, output_size))

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        out = x
        for i in range(self.n_layers-1):
            out = self.MLP[i](self.drop(out))
            out = self.activation_function(out)
        return self.MLP[-1](self.drop(out))
    
    
# class NeuralNet(nn.Module):
#     def __init__(self, out_sz, sizes, drops, y_range=None, use_bn=False, f=F.relu)
#     def __init__(self, input_size, mid_size, output_size, dropout):
#         super().__init__()
        
#         self.linear = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
#         self.bns = nn.ModuleList([nn.BatchNorm1d(size) for size in sizes[1:]])
#         for layer in self.linear:
#             kaiming_normal(layer.weight.data)
#         self.dropout = [nn.Dropout(drop) for drop in drops]
#         self.output = nn.Linear(sizes[-1], 1)
#         kaiming_normal(self.output.weight.data)
#         self.f = f
#         self.use_bn = use_bn
            
        
#     def forward(self, X):
#         for linear, drop, norm in zip(self.linear, self.dropout, self.bns):
#             X = self.f(linear(X))
#             if self.use_bn: 
#                 X = norm(X)
#             X = drop(X)
#         X = self.output(X)