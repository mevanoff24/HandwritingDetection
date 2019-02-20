import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal
from torch import tensor
import numpy as np
import math

# from models.context2vec.src.negative_sampling import NegativeSampling
# from models.context2vec.src.utils import init_embeddings

def init_embeddings(x):
    """Initialize embedding weights uniform values"""
    x = x.weight.data
    value = 2 / (x.size(1) + 1)
    x.uniform_(-value, value)

def create_embedding_layer(vocab_size, word_embed_size, pad_idx):
    """Defines embedding layer"""
    return nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=word_embed_size,
                                    padding_idx=pad_idx)

def create_rnn_layer(word_embed_size, hidden_size, n_layers, batch_first, layer_type=nn.LSTM):
    """Defines RNN layer"""
    return layer_type(input_size=word_embed_size,
                               hidden_size=hidden_size,
                               num_layers=n_layers,
                               batch_first=batch_first) 

class Context2vec(nn.Module):
    """
    Context2vec model 
    
    Attributes:
        vocab_size (int): Vocabulary size
        word_embed_size (int): Size of word embedding layer
        hidden_size (int): Size of hidden layer
        n_layers (int): Number of layers for LSTM cells
        device (int): Device id for GPU
        inference (boolean): Prediction time or training time
        rnn_output_size (int): Size of output from RNN
        l2r_emb (object): left context pyTorch Embedding module 
        r2l_emb (object): right context pyTorch Embedding module
        l2r_rnn (object): left context pyTorch RNN module
        r2l_rnn (object): right context pyTorch RNN module 
        dropout (object): pyTorch Dropout module
        criterion (object): NegativeSampling class object
        MLP (object): Top Model FC Neural Network

    """
    def __init__(self, vocab_size, counter, word_embed_size, hidden_size, n_layers, bidirectional, dropout,
                 pad_idx, device, inference):
        """
        Args:
            counter (list): Frequency of words in text dataset
            bidirectional (boolean): Use bidirectional RNN if True
            pad_idx (int): Index of padding token
        """

        super().__init__()
        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.inference = inference
        self.rnn_output_size = hidden_size
        
        # embedding
        self.l2r_emb = create_embedding_layer(vocab_size, word_embed_size, pad_idx)
        self.r2l_emb = create_embedding_layer(vocab_size, word_embed_size, pad_idx)
        for embed in [self.l2r_emb, self.r2l_emb]:
            init_embeddings(embed)
        # rnn
        self.l2r_rnn = create_rnn_layer(word_embed_size, hidden_size, n_layers, batch_first=True)
        self.r2l_rnn = create_rnn_layer(word_embed_size, hidden_size, n_layers, batch_first=True)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # loss
        self.criterion = NegativeSampling(hidden_size, counter, pad_idx=pad_idx, num_neg=10, power=0.75,
                                          device=device) # num_neg=10, power=0.75 used in paper
        
        self.MLP = NeuralNet(input_size=hidden_size*2, mid_size=hidden_size*2, output_size=hidden_size,
                                                               dropout=dropout)
        
    def forward(self, X, y, target_pos=None):
        """Forward pass of model"""
        batch_size, seq_len = X.size()
        X_reversed = X.flip(1)[:, :-1]
        X = X[:, :-1]
        
        left2right_embed = self.l2r_emb(X)
        right2left_embed = self.r2l_emb(X_reversed)
        
        left2right_out, _ = self.l2r_rnn(left2right_embed)
        right2left_out, _ = self.r2l_rnn(right2left_embed)
        
        left2right_out = left2right_out[:, :-1, :]
        right2left_out = right2left_out[:, :-1, :].flip(1)
        # TESTING
        if self.inference:
            left2right_out = left2right_out[0, target_pos]
            right2left_out = right2left_out[0, target_pos]
            out = self.MLP(torch.cat((left2right_out, right2left_out), dim=0))
            return out
        # TRAINING 
        else:
            out = self.MLP(torch.cat((left2right_out, right2left_out), dim=2)) # dim = 2
            loss = self.criterion(y, out)
            return loss 
        
    def run_inference(self, input_tokens, target, target_pos, k=10):
        """Prediction process for model"""
        context_vector = self.forward(input_tokens, y=None, target_pos=target_pos)
        if target is None:
            topv, topi = ((self.criterion.W.weight*context_vector).sum(dim=1)).data.topk(k)
            return topv, topi
        else:
            context_vector /= torch.norm(context_vector, p=2)
            target_vector = self.criterion.W.weight[target]
            target_vector /= torch.norm(target_vector, p=2)
            similarity = (target_vector * context_vector).sum()
            return similarity.item()
        
    def norm_embedding_weight(self, embeddings):
        """Normalize embedding weights"""
        embeddings.weight.data /= torch.norm(embeddings.weight.data, p=2, dim=1, keepdim=True)
        embeddings.weight.data[embeddings.weight.data != embeddings.weight.data] = 0

        
    
        
class NeuralNet(nn.Module):
    """
    Fully Connected top layer model for Context2vec

    Attributes:
        input_size (int): Input size for model
        mid_size (int): Hidden layer size 
        output_size (int): Output_size for model
        n_layers (int): Number of layers for LSTM cells
        drop (object): pyTorch Dropout module
        MLP (object): ModuleList for this model
        activation_function (str): Non-linear activation function
    """
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
    
    
    
class NegativeSampling(nn.Module):
    """
    Negative Sampling Loss

    Attributes:
        counter (list): Frequency of words in text dataset
        num_neg (int): Number of negative samples to draw
        power (float): Power to decrease the probability for more frequent words
        device (int): Device id for GPU
        W (object): pyTorch Embedding module
        log_loss: (object): pyTorch Log Sigmoid module
        sampler (object): WalkerAlias class

    """
    def __init__(self, embed_size, counter, num_neg, power, device, pad_idx):
        """
        Args:
            embed_size (int): Size of word embedding layer
            pad_idx (int): Index of padding token
        """
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
        """Negative Sample to tensor"""
        return tensor(self.sampler.sample(shape=shape), dtype=torch.long, device=self.device)
    
    def forward(self, X, context):
        """Forward pass for Negative Sampling"""
        batch_size, seq_len = X.size()
        embedding = self.W(X)
        pos_loss = self.log_loss((embedding * context).sum(2))

        neg_samples = self.negative_sampling(shape=(batch_size, seq_len, self.num_neg))
        neg_embedding = self.W(neg_samples)
        neg_loss = self.log_loss((-neg_embedding * context.unsqueeze(2)).sum(3)).sum(2)
        return -(pos_loss + neg_loss).sum()

        

'''
The class WalkerAlias is taken from the original author of Context2vec
https://github.com/chainer/chainer/blob/v5.2.0/chainer/utils/walker_alias.py
'''

import numpy

class WalkerAlias(object):
    """Implementation of Walker's alias method.
    This method generates a random sample from given probabilities
    :math:`p_1, \\dots, p_n` in :math:`O(1)` time.
    It is more efficient than :func:`~numpy.random.choice`.
    This class works on both CPU and GPU.
    Args:
        probs (float list): Probabilities of entries. They are normalized with
                            `sum(probs)`.
    See: `Wikipedia article <https://en.wikipedia.org/wiki/Alias_method>`_
    """

    def __init__(self, probs):
        prob = numpy.array(probs, numpy.float32)
        prob /= numpy.sum(prob)
        threshold = numpy.ndarray(len(probs), numpy.float32)
        values = numpy.ndarray(len(probs) * 2, numpy.int32)
        il, ir = 0, 0
        pairs = list(zip(prob, range(len(probs))))
        pairs.sort()
        for prob, i in pairs:
            p = prob * len(probs)
            while p > 1 and ir < il:
                values[ir * 2 + 1] = i
                p -= 1.0 - threshold[ir]
                ir += 1
            threshold[il] = p
            values[il * 2] = i
            il += 1
        # fill the rest
        for i in range(ir, len(probs)):
            values[i * 2 + 1] = 0

        assert((values < len(threshold)).all())
        self.threshold = threshold
        self.values = values
        self.use_gpu = False

    def to_gpu(self):
        """Make a sampler GPU mode.
        """
        if not self.use_gpu:
            self.threshold = cuda.to_gpu(self.threshold)
            self.values = cuda.to_gpu(self.values)
            self.use_gpu = True

    def to_cpu(self):
        """Make a sampler CPU mode.
        """
        if self.use_gpu:
            self.threshold = cuda.to_cpu(self.threshold)
            self.values = cuda.to_cpu(self.values)
            self.use_gpu = False

    def sample(self, shape):
        """Generates a random sample based on given probabilities.
        Args:
            shape (tuple of int): Shape of a return value.
        Returns:
            Returns a generated array with the given shape. If a sampler is in
            CPU mode the return value is a :class:`numpy.ndarray` object, and
            if it is in GPU mode the return value is a :class:`cupy.ndarray`
            object.
        """
        if self.use_gpu:
            return self.sample_gpu(shape)
        else:
            return self.sample_cpu(shape)

    def sample_cpu(self, shape):
        ps = numpy.random.uniform(0, 1, shape)
        pb = ps * len(self.threshold)
        index = pb.astype(numpy.int32)
        left_right = (self.threshold[index] < pb - index).astype(numpy.int32)
        return self.values[index * 2 + left_right]

    def sample_gpu(self, shape):
        ps = cuda.cupy.random.uniform(size=shape, dtype=numpy.float32)
        vs = cuda.elementwise(
            'T ps, raw T threshold , raw S values, int32 b',
            'int32 vs',
            '''
            T pb = ps * b;
            int index = __float2int_rd(pb);
            // fill_uniform sometimes returns 1.0, so we need to check index
            if (index >= b) {
              index = 0;
            }
            int lr = threshold[index] < pb - index;
            vs = values[index * 2 + lr];
            ''',
            'walker_alias_sample'
        )(ps, self.threshold, self.values, len(self.threshold))
        return vs
    
