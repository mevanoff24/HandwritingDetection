import numpy as np
import os
import time
import torch
from torch import optim
import boto3
from io import BytesIO

from src.mscc_eval import mscc_evaluation
from src.model import Context2vec
from src.negative_sampling import NegativeSampling
from src.utils import write_embedding, write_config, read_config, load_vocab
from src.dataset import WikiDataset
from src.args import parse_args
from src.config import Config



def main(train=True):
    args = parse_args()
    config = Config(args.config_file)
    gpu_id = args.gpu
    use_cuda = torch.cuda.is_available() and gpu_id > -1
    max_sent_length = 64
    if use_cuda:
        device = torch.device('cuda:{}'.format(gpu_id))
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')

    if train:        
        if args.use_s3 == 'true':
            S3_BUCKET = 'handwrittingdetection'
            S3_WIKI_TRAIN_PATH = 'data/wiki_train/rawwikitext-2-train.npy'
            S3_WIKI_VAL_PATH = 'data/wiki_valid/rawwikitext-2-valid.npy'
            print('Loading Training Data from S3 bucket {} -- {}'.format(S3_BUCKET, S3_WIKI_TRAIN_PATH))
            client = boto3.resource('s3')
            bucket = client.Bucket(S3_BUCKET)
            sentences = np.load(BytesIO(bucket.Object(S3_WIKI_TRAIN_PATH).get()['Body'].read()))
        else:
            sentences = np.load(args.train_file)

        print('Creating dataset')
        dataset = WikiDataset(sentences, config.batch_size, config.min_freq, device)
        counter = np.array([dataset.vocab.freqs[word] if word in dataset.vocab.freqs else 0
                            for word in dataset.vocab.itos])
        if args.load == 'true':
            # load saved model
            config_file = args.modelfile+'.config.json'
            config_dict = read_config(config_file)
            model = Context2vec(vocab_size=config_dict['vocab_size'],
                            counter=[1]*config_dict['vocab_size'],
                            word_embed_size=config_dict['word_embed_size'],
                            hidden_size=config_dict['hidden_size'],
                            n_layers=config_dict['n_layers'],
                            bidirectional=config_dict['bidirectional'],
                            dropout=config_dict['dropout'],
                            pad_idx=config_dict['pad_index'],
                            device=device,
                            inference=False).to(device)
            model.load_state_dict(torch.load(args.modelfile, map_location=device))
            optimizer = optim.Adam(model.parameters(), lr=config_dict['learning_rate'])
            optimizer.load_state_dict(torch.load(args.modelfile+'.optim'))
#             itos, stoi = load_vocab(args.wordsfile)
#             unk_token = config_dict['unk_token']
#             bos_token = config_dict['bos_token']
#             eos_token = config_dict['eos_token']
            
#             dataset = WikiDataset(sentences, config.batch_size, config.min_freq, device, itos=itos, stoi=stoi,
#                                   unk_token=unk_token, bos_token=bos_token, eos_token=eos_token)
#             val_dataset = WikiDataset(val_sentences, config.batch_size, config.min_freq, device, train_loader=dataset, 
#                                       itos=itos, stoi=stoi, unk_token=unk_token, bos_token=bos_token, eos_token=eos_token,
#                                       device=device)
        
            best_val_score = int(config_dict['best_val_score'])
            best_model = config_dict['best_model']
            print('Saved model loaded')
        
        else:
            # train from scratch
            model = Context2vec(vocab_size=len(dataset.vocab),
                            counter=counter,
                            word_embed_size=config.word_embed_size,
                            hidden_size=config.hidden_size,
                            n_layers=config.n_layers,
                            bidirectional=True,
                            dropout=config.dropout,
                            pad_idx=dataset.pad_idx,
                            device=device,
                            inference=False).to(device)
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            best_val_score = float('inf')
            best_model = None

        print('batch_size:{}, n_epochs:{}, word_embed_size:{}, hidden_size:{}, device:{}'.format(
                             config.batch_size, config.n_epochs, config.word_embed_size, config.hidden_size, device))
        print(model)
        
        if args.val_file:
            val_sentences = np.load(args.val_file)
                
        if args.use_s3 == 'true':
            print('Loading Validation Data from S3 bucket {} -- {}'.format(S3_BUCKET, S3_WIKI_VAL_PATH))
            val_sentences = np.load(BytesIO(bucket.Object(S3_WIKI_VAL_PATH).get()['Body'].read()))
            
        if args.use_s3 == 'true' or args.val_file:
            val_dataset = WikiDataset(val_sentences, config.batch_size, config.min_freq, 
                                                                      train_loader=dataset, device=device)
            print('Creating Validation dataset')
            
            
        log_dir_name = 'logs'    
        log_dir = os.path.dirname(log_dir_name)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        iterations_without_improvement = 0
        print('Training Begins')
        interval = 1e6
        for epoch in range(config.n_epochs):
            begin_time = time.time()
            cur_at = begin_time
            total_loss = 0.0
            val_total_loss = 0.0
            word_count = 0
            next_count = interval
            last_accum_loss = 0.0
            last_word_count = 0
            
            model.train() 
            for iterator in dataset.get_batch_iter(config.batch_size):
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
            if args.use_s3 == 'true' or args.val_file:
                model.eval()
                for val_iterator in val_dataset.get_batch_iter(config.batch_size):
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
                # update best model 
                if val_total_loss < best_val_score:
                    best_val_score = val_total_loss
                    print(best_val_score.cpu().numpy())
                    
                    best_model = epoch + 1
                    print('Saving Best model to {}. Epoch Number {}'.format(args.wordsfile.split('/')[0], best_model))
                    iterations_without_improvement = 0
                    
                    # shutil.rmtree(args.wordsfile.split('/')[0], ignore_errors=True)
                    output_dir = os.path.dirname(args.wordsfile)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # save model if improvement 
                    write_embedding(dataset.vocab.itos, model.criterion.W, use_cuda, args.wordsfile)
                    torch.save(model.state_dict(), args.modelfile)
                    torch.save(optimizer.state_dict(), args.modelfile+'.optim')
                    output_config_file = args.modelfile+'.config.json'
                    write_config(output_config_file,
                                 vocab_size=len(dataset.vocab),
                                 word_embed_size=config.word_embed_size,
                                 hidden_size=config.hidden_size,
                                 n_layers=config.n_layers,
                                 bidirectional=True,
                                 dropout=config.dropout,
                                 pad_index=dataset.pad_idx,
                                 unk_token=dataset.unk_token,
                                 bos_token=dataset.bos_token,
                                 eos_token=dataset.eos_token,
                                 learning_rate=config.learning_rate,
                                 best_val_score=str(best_val_score),
                                 best_model=best_model)
                else:
                    iterations_without_improvement += 1

            # ---------
            with open(os.path.join(log_dir_name, args.log_filename), 'a') as f:
                if args.val_file or args.use_s3 == 'true':
                    val_out = val_total_loss.item()
                else:
                    val_out = ''  
                f.write(str(epoch) + ' ' + str(total_loss.item()) + ' ' + str(val_out) + '\n')
            # early stopping 
            if iterations_without_improvement >= args.patience:   
                print('Best Epoch Number: {}'.format(best_model))
                return
        
        # No validation set -- Save model after final epoch
        if args.use_s3 == 'false' and not args.val_file:
            print('Saving Final Model')
            output_dir = os.path.dirname(args.wordsfile)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            write_embedding(dataset.vocab.itos, model.criterion.W, use_cuda, args.wordsfile)
            torch.save(model.state_dict(), args.modelfile)
            torch.save(optimizer.state_dict(), args.modelfile+'.optim')
            output_config_file = args.modelfile+'.config.json'
            write_config(output_config_file,
                         vocab_size=len(dataset.vocab),
                         word_embed_size=config.word_embed_size,
                         hidden_size=config.hidden_size,
                         n_layers=config.n_layers,
                         bidirectional=True,
                         dropout=config.dropout,
                         pad_index=dataset.pad_idx,
                         unk_token=dataset.unk_token,
                         bos_token=dataset.bos_token,
                         eos_token=dataset.eos_token,
                         learning_rate=config.learning_rate,
                         best_val_score=str(best_val_score),
                         best_model=best_model)

                
if __name__ == '__main__':
    main()