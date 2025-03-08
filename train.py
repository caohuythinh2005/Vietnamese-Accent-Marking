import re
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import pandas as pd 
import os 
import math
import torchtext.data
import torchtext.datasets
import torchtext.vocab
from config import opt
from collections import Counter
from torch.utils.data import DataLoader
from model import *

device = torch.device('cuda')

def remove_tone_line(utf8_str):
    intab_l = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ"
    intab_u = "ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    intab = list(intab_l+intab_u)

    outtab_l = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d"
    outtab_u = "A"*17 + "O"*17 + "E"*11 + "U"*11 + "I"*5 + "Y"*5 + "D"
    outtab = outtab_l + outtab_u
    # Khởi tạo regex tìm kiếm các vị trí nguyên âm có dấu 'ạ|ả|ã|...'
    r = re.compile("|".join(intab))

    # Dictionary có key-value là từ có dấu-từ không dấu. VD: {'â' : 'a'}
    replaces_dict = dict(zip(intab, outtab))
    # Thay thế các từ có dấu xuất hiện trong tìm kiếm của regex bằng từ không dấu tương ứng
    non_dia_str = r.sub(lambda m: replaces_dict[m.group(0)], utf8_str)
    return non_dia_str

class tokenize(object):
    def __init__(self, param : str) -> None:
        self.param = param

    def tokenizer(self, sentence):
        if self.param == "with_accents":
            tokens = re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)
            return tokens
        
        if self.param ==  "without_accents":
            sentence_ipt = remove_tone_line(sentence)
            tokens = re.findall(r'\w+|[^\w\s]', sentence_ipt, re.UNICODE)
            return tokens

specials = ['<unk>', '<pad>', '<sos>', '<eos>']

def load_dataset(config):
    print('Loading dataset...')
    tokenize_opt = tokenize(config["opt"])
    tokenize_ipt = tokenize(config["ipt"])
    train_dataset_ipt = []
    train_dataset_opt = []
    val_dataset_ipt = []
    val_dataset_opt = []
    test_dataset_ipt = []
    test_dataset_opt = []
    counter_opt = Counter()
    counter_ipt = Counter()
    counter_opt.update(specials)
    counter_ipt.update(specials)
    with open(config['filename'], 'r', encoding='utf-8') as f: 
        for i in tqdm(range(config['max_len_load'])):
            line = f.readline()
            [_, origin_seq] = line.split('\t')
            line_opt = tokenize_opt.tokenizer(origin_seq)
            line_ipt = tokenize_ipt.tokenizer(origin_seq)
            line_opt = line_opt[:(config['seq_len']-2)]
            line_ipt = line_ipt[:(config['seq_len']-2)]
            counter_opt.update(line_opt)
            counter_ipt.update(line_ipt)
            if i < config['train_size']:           
                train_dataset_opt.append(line_opt)
                train_dataset_ipt.append(line_ipt)
            elif i < config['train_size'] + config['val_size']:
                val_dataset_opt.append(line_opt)
                val_dataset_ipt.append(line_ipt)
            else:
                test_dataset_opt.append(line_opt)
                test_dataset_ipt.append(line_ipt)
    f.close()

    vocab_opt = torchtext.vocab.Vocab(counter_opt, min_freq=1)
    vocab_ipt = torchtext.vocab.Vocab(counter_ipt, min_freq=1)
    return train_dataset_ipt, train_dataset_opt, val_dataset_ipt, val_dataset_opt, test_dataset_ipt, test_dataset_opt, tokenize_ipt, tokenize_opt, vocab_ipt, vocab_opt


train_dataset_ipt, train_dataset_opt, val_dataset_ipt, val_dataset_opt, test_dataset_ipt, test_dataset_opt, tokenize_ipt, tokenize_opt, vocab_ipt, vocab_opt = load_dataset(opt)


def encode(x, vocab):
    return [vocab.stoi[s] for s in x]

def decode(x, vocab):
    return [vocab.itos[s] for s in x]

def causal_mask(size):
    '''
    mask được sử dụng cho quá trình dự đoán, mô hình không thấy được tương lai
    '''
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int64).to(device)
    return mask == 0

from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, train_dataset_ipt, train_dataset_opt, vocab_ipt, vocab_opt, config):
        super().__init__()
        self.seq_len = config['seq_len']

        self.train_dataset_ipt = train_dataset_ipt
        self.train_dataset_opt = train_dataset_opt

        self.vocab_ipt = vocab_ipt
        self.vocab_opt = vocab_opt

        self.sos_token_ipt = torch.tensor([vocab_ipt.stoi['<sos>']], dtype=torch.int64, device=device)
        self.eos_token_ipt = torch.tensor([vocab_ipt.stoi['<eos>']], dtype=torch.int64, device=device)
        self.pad_token_ipt = torch.tensor([vocab_ipt.stoi['<pad>']], dtype=torch.int64, device=device)

        self.sos_token_opt = torch.tensor([vocab_opt.stoi['<sos>']], dtype=torch.int64, device=device)
        self.eos_token_opt = torch.tensor([vocab_opt.stoi['<eos>']], dtype=torch.int64, device=device)
        self.pad_token_opt = torch.tensor([vocab_opt.stoi['<pad>']], dtype=torch.int64, device=device)

    def __len__(self):
        return len(self.train_dataset_ipt)
    
    def __getitem__(self, index):
        ipt_tokenized = train_dataset_ipt[index]
        opt_tokenized = train_dataset_opt[index]

        enc_num_padding_tokens = self.seq_len - len(ipt_tokenized) - 2
        dec_num_padding_tokens = self.seq_len - len(opt_tokenized) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long! Try to increase seq_len in config.py")
        
        encoder_input = torch.cat(
            [
                self.sos_token_ipt,
                torch.tensor(encode(ipt_tokenized, self.vocab_ipt), dtype=torch.int64, device=device),
                self.eos_token_ipt,
                torch.tensor([self.pad_token_ipt] * enc_num_padding_tokens, dtype=torch.int64, device=device)
            ],
            dim=0
        )

        decoder_input = torch.cat(
            [
                self.sos_token_opt,
                torch.tensor(encode(opt_tokenized, self.vocab_opt), dtype=torch.int64, device=device),
                torch.tensor([self.pad_token_opt] * dec_num_padding_tokens, dtype=torch.int64, device=device)
            ],
            dim=0
        )

        label = torch.cat(
            [
                torch.tensor(encode(opt_tokenized, self.vocab_opt), dtype=torch.int64, device=device),
                self.eos_token_opt,
                torch.tensor([self.pad_token_opt] * dec_num_padding_tokens, dtype=torch.int64, device=device)
            ],
            dim=0
        )
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token_ipt).unsqueeze(0).unsqueeze(0).long().to(device), # (1, seq_len) do input (batch, seq_len, d_model)
            "decoder_mask": (decoder_input != self.pad_token_opt).unsqueeze(0).long().to(device) & causal_mask(decoder_input.size(0)), # (seq_len, seq_len)
            "label": label, # (seq_len)
            # "ipt_tokenized": ipt_tokenized, 
            # "opt_tokenized": opt_tokenized
        }
    
def get_ds(config):
    train_ds = CustomDataset(train_dataset_ipt=train_dataset_ipt,
                             train_dataset_opt=train_dataset_opt,
                             vocab_ipt=vocab_ipt,
                             vocab_opt=vocab_opt,
                             config=config)
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)

    return train_dataloader

def get_model(config, ipt_vocab_len, opt_vocab_len):
    model = build_transformer(ipt_vocab_len, opt_vocab_len, config['seq_len'], config['seq_len'])
    return model

ipt_vocab_len = len(vocab_ipt.stoi)
opt_vocab_len = len(vocab_opt.stoi)

def train_model(config):
    train_dataloader= get_ds(config)
    model = get_model(config, ipt_vocab_len=ipt_vocab_len, opt_vocab_len=opt_vocab_len).to(device)
    state = torch.load('model.pt')
    model.load_state_dict(state['model_state_dict'])
    print("Using device: ", device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab_opt.stoi['<pad>'], label_smoothing=0.1).to(device)

    for epoch in range(0, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing Epoch {epoch:02d}')
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, opt_vocab_len), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'model.pt')

print(device)
if __name__ ==  '__main__':

    config = opt
    train_model(config)