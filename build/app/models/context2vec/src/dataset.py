import torch
import torch.nn as nn
import numpy as np
from torchtext.data import Field, Dataset, Example, Iterator


class WikiDataset:
    """
    Dataset Class for Language Model 

    Attributes:
        sent_dict: (dict): Dictionary ordering input text by length
        pad_token (str): Padding token character
        unk_token (str): Unknown token character
        bos_token (str): Beginning of string token character
        eos_token (str): End of sentence token character
        device (int): Device id for GPU
        sentence_field (Object): pyTorch Field Object with input text
        sentence_field_id (Object): pyTorch Field Object with input id
        vocab (dict): Vocab for dataset
        pad_idx (int): Index value for padding 
    """
    def __init__(self, X, batch_size, min_freq, device, train_loader=None, pad_token='<PAD>', unk_token='<UNK>', 
                                      bos_token='<BOS>', eos_token='<EOS>', seed=100):  
        """
        Args:
            X (list): Input text 
            batch_size (int): Size of training dataset per iteration
            seed (int): Random State for reproducible results
        """
        super().__init__() 
        np.random.seed(seed)
        
        self.is_training = True
        if train_loader:
            self.is_training = False
        
        self.sent_dict = self._gathered_by_lengths(X)
        self.device = device
        # set up torchtext Fields if training set
        if self.is_training:
            self.pad_token = pad_token
            self.unk_token = unk_token
            self.bos_token = bos_token
            self.eos_token = eos_token
            self.sentence_field = Field(use_vocab=True, unk_token=self.unk_token, pad_token=self.pad_token,
                                         init_token=self.bos_token, eos_token=self.eos_token,
                                         batch_first=True, include_lengths=False)
            self.sentence_field_id = Field(use_vocab=False, batch_first=True)
            # build vocab
            self.sentence_field.build_vocab(X, min_freq=min_freq)
            self.vocab = self.sentence_field.vocab
        else:
        # validation set
            self.pad_token = train_loader.pad_token
            self.unk_token = train_loader.unk_token
            self.bos_token = train_loader.bos_token
            self.eos_token = train_loader.eos_token
            self.sentence_field = train_loader.sentence_field
            self.sentence_field_id = train_loader.sentence_field_id
            self.vocab = train_loader.vocab
             
        if self.pad_token: self.pad_idx = self.sentence_field.vocab.stoi[self.pad_token]
        self.dataset = self._create_dataset(self.sent_dict, X)
    
    
    def get_raw_sentence(self, X):
        """Reverse mapping from int to string"""
        return [[self.vocab.itos[idx] for idx in sentence] for sentence in X]
     
        
    def _gathered_by_lengths(self, X):
        """Order input text by length of text"""
        lengths = [(index, len(sent)) for index, sent in enumerate(X)]
        lengths = sorted(lengths, key=lambda x: x[1], reverse=True)

        sent_dict = {}
        current_length = -1
        for i, length in lengths:
            if current_length == length:
                sent_dict[length].append(i)
            else:
                sent_dict[length] = [i]
                current_length = length

        return sent_dict
    
    
    def _create_dataset(self, sent_dict, X):
        """Create torchtext dataset"""
        datasets = {}
        _fields = [('sentence', self.sentence_field),
                   ('id', self.sentence_field_id)]
        for length, index in sent_dict.items():
            index = np.array(index)
            items = [*zip(X[index], index[:, np.newaxis])]
            datasets[length] = Dataset(self._get_examples(items, _fields), _fields)
        if self.is_training:
            out = np.random.permutation(list(datasets.values()))
        else:
            out = list(datasets.values())
        return out
    
    
    def _get_examples(self, items, fields):
        """Defines a single training example. Stores each column of the example as an attribute."""
        return [Example.fromlist(item, fields) for item in items]

    
    def get_batch_iter(self, batch_size):
        """Iterator for dataset"""
        def sort(data):
            return len(getattr(data, 'sentence'))

        for dataset in self.dataset:
            yield Iterator(dataset=dataset,
                                batch_size=batch_size,
                                sort_key=sort,
                                train=self.is_training,
                                repeat=False,
                                device=self.device)            