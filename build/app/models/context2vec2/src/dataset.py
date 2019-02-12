import torch
import torch.nn as nn
import numpy as np
from torchtext.data import Field, Dataset, Example, Iterator


class WikiDataset:
    def __init__(self, X, batch_size, min_freq, device, pad_token='<PAD>', unk_token='<UNK>', 
                                      bos_token='<BOS>', eos_token='<EOS>', seed=100):
        super().__init__() 
        np.random.seed(seed)
        self.sent_dict = self._gathered_by_lengths(X)
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.device = device
        # set up torchtext Fields
        self.sentence_field = Field(use_vocab=True, unk_token=self.unk_token, pad_token=self.pad_token,
                                         init_token=self.bos_token, eos_token=self.eos_token,
                                         batch_first=True, include_lengths=False)
        self.sentence_field_id = Field(use_vocab=False, batch_first=True)
        # build vocal
        self.sentence_field.build_vocab(X, min_freq=min_freq)
        self.vocab = self.sentence_field.vocab
        if self.pad_token: self.pad_idx = self.sentence_field.vocab.stoi[self.pad_token]
        self.dataset = self._create_dataset(self.sent_dict, X)
    
    def get_raw_sentence(self, X):
        return [[self.vocab.itos[idx] for idx in sentence] for sentence in X]   
     
        
    def _gathered_by_lengths(self, X):
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
        datasets = {}
        _fields = [('sentence', self.sentence_field),
                   ('id', self.sentence_field_id)]
        for length, index in sent_dict.items():
            index = np.array(index)
            items = [*zip(X[index], index[:, np.newaxis])]
            datasets[length] = Dataset(self._get_examples(items, _fields), _fields)
        return np.random.permutation(list(datasets.values()))
    
    
    def _get_examples(self, items, fields):
        return [Example.fromlist(item, fields) for item in items]

    
    def get_batch_iter(self, batch_size):

        def sort(data):
            return len(getattr(data, 'sentence'))

        for dataset in self.dataset:
            yield Iterator(dataset=dataset,
                                batch_size=batch_size,
                                sort_key=sort,
                                train=True,
                                repeat=False,
                                device=self.device)