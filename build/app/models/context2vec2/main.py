import numpy as np
import os
import time
import torch
from torch import optim
from src.mscc_eval import mscc_evaluation
from src.model import Context2vec
from src.negative_sampling import NegativeSampling
from src.utils import write_embedding, write_config 
from src.dataset import WikiDataset
# from src.util.config import Config
# from src.util.io import write_embedding, write_config, read_config, load_vocab
import boto3
from io import BytesIO






train = True
word_embed_size = 300
hidden_size = 300
n_layers = 1
dropout = 0.00
n_epochs = 1
batch_size = 100
min_freq = 1
ns_power = 0.75
learning_rate = 1e-4
gpu_id = 0

def main(train_path):
#     use_cuda = torch.cuda.is_available()
    use_cuda = torch.cuda.is_available() and gpu_id > -1
    max_sent_length = 64
    if use_cuda:
        device = torch.device('cuda:{}'.format(gpu_id))
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')

    if train:
#         batch_size = batch_size
#         n_epochs = n_epochs
#         word_embed_size = word_embed_size
#         hidden_size = hidden_size
#         learning_rate = learning_rate

        
        if use_s3:
            print('Loading Training Data from S3 bucket {} -- {}'.format(S3_BUCKET, S3_WIKI_TRAIN_PATH))
            client = boto3.resource('s3')
            bucket = client.Bucket(S3_BUCKET)
            sentences = np.load(BytesIO(bucket.Object(S3_WIKI_TRAIN_PATH).get()['Body'].read()))
        else:
            sentences = np.load(train_path)
            
        
        
        print('Creating dataset')
        dataset = WikiDataset(sentences, batch_size, min_freq, device)
        counter = np.array([dataset.vocab.freqs[word] if word in dataset.vocab.freqs else 0
                            for word in dataset.vocab.itos])
        model = Context2vec(vocab_size=len(dataset.vocab),
                            counter=counter,
                            word_embed_size=word_embed_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            bidirectional=True,
                            dropout=dropout,
                            pad_idx=dataset.pad_idx,
                            device=device,
                            inference=False).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print('batch_size:{}, n_epochs:{}, word_embed_size:{}, hidden_size:{}, device:{}'.format(
                                                batch_size, n_epochs, word_embed_size, hidden_size, device))
        print(model)
        
        if use_validation_set:
            if use_s3:
                print('Loading Validation Data from S3 bucket {} -- {}'.format(S3_BUCKET, S3_WIKI_VAL_PATH))
                val_sentences = np.load(BytesIO(bucket.Object(S3_WIKI_VAL_PATH).get()['Body'].read()))
            else:
                val_sentences = np.load(validation_data)
            
            print('Creating Validation dataset')
            val_dataset = WikiDataset(val_sentences, batch_size, min_freq, device)
            val_counter = np.array([val_dataset.vocab.freqs[word] if word in val_dataset.vocab.freqs else 0
                                for word in val_dataset.vocab.itos])
            
        log_dir = os.path.dirname(log_dir_name)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
            
        best_val_score = float('inf')
        print('Training Begins')
        interval = 1e6
        for epoch in range(n_epochs):
            begin_time = time.time()
            cur_at = begin_time
            total_loss = 0.0
            val_total_loss = 0.0
            word_count = 0
            next_count = interval
            last_accum_loss = 0.0
            last_word_count = 0
            
            model.train() 
            for iterator in dataset.get_batch_iter(batch_size):
                for batch in iterator:
                    sentence = getattr(batch, 'sentence')
                    target = sentence[:, 1:-1]
                    if target.size(0) == 0:
                        continue
                    optimizer.zero_grad()
                    loss = model(sentence, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.data.mean()

                    minibatch_size, sentence_length = target.size()
                    word_count += minibatch_size * sentence_length
                    accum_mean_loss = float(total_loss)/word_count if total_loss > 0.0 else 0.0
                    if word_count >= next_count:
                        now = time.time()
                        duration = now - cur_at
                        throuput = float((word_count-last_word_count)) / (now - cur_at)
                        cur_mean_loss = (float(total_loss)-last_accum_loss)/(word_count-last_word_count)
                        print('{} words, {:.2f} sec, {:.2f} words/sec, {:.4f} accum_loss/word, {:.4f} cur_loss/word'
                              .format(word_count, duration, throuput, accum_mean_loss, cur_mean_loss))
                        next_count += interval
                        cur_at = now
                        last_accum_loss = float(total_loss)
                        last_word_count = word_count


            
            # ---------
            # VAL PHASE
            model.eval()
            for val_iterator in val_dataset.get_batch_iter(batch_size):
                with torch.no_grad():
                    for batch in val_iterator:
                        val_sentence = getattr(batch, 'sentence')
                        val_target = val_sentence[:, 1:-1]
                        if val_target.size(0) == 0:
                            continue
                        val_loss = model(val_sentence, val_target)
                        val_total_loss += val_loss.data.mean()
            print('Train loss: {} -- Valid loss: {}'.format(total_loss.item(), val_total_loss.item()))
            print()
            
    
        # ---------
            with open(os.path.join(log_dir_name, log_filename), 'a') as f:
                f.write(str(epoch) + ' ' + str(total_loss.item()) + ' ' + str(val_total_loss.item()) + '\n')
                
                
        output_dir = os.path.dirname(wordsfile)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        write_embedding(dataset.vocab.itos, model.neg_sample_loss.W, use_cuda, wordsfile)
        torch.save(model.state_dict(), modelfile)
        torch.save(optimizer.state_dict(), modelfile+'.optim')
        output_config_file = modelfile+'.config.json'
        write_config(output_config_file,
                     vocab_size=len(dataset.vocab),
                     word_embed_size=word_embed_size,
                     hidden_size=hidden_size,
                     n_layers=n_layers,
                     bidirectional=True,
                     dropout=dropout,
                     pad_index=dataset.pad_idx,
                     unk_token=dataset.unk_token,
                     bos_token=dataset.bos_token,
                     eos_token=dataset.eos_token,
                     learning_rate=learning_rate)   
                
                
if __name__ == '__main__':
    train_path = '../../../../data/processed/rawwikitext-2-train.npy'
    array_file = True  
    text_file = False
    use_s3 = False
    use_validation_set = True
    # validation_data = '../../../../data/processed/rawwikitext-2-valid.npy'
    validation_data = '../../../../data/processed/rawwikitext-2-valid.npy'
    S3_BUCKET = 'handwrittingdetection'
    S3_WIKI_TRAIN_PATH = 'data/wiki_train/rawwikitext-2-train.npy'
    S3_WIKI_VAL_PATH = 'data/wiki_valid/rawwikitext-2-valid.npy'
    wordsfile = 'models/embedding.vec'
    modelfile = 'models/model.param'
    log_dir_name = 'logs'
    log_filename = 'log_dir1.txt'
    
    main(train_path)