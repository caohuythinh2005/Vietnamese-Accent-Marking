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
from pathlib import Path
import sys

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
    
    

    vocab_opt = torchtext.vocab.Vocab(counter_opt, min_freq=1)
    vocab_ipt = torchtext.vocab.Vocab(counter_ipt, min_freq=1)
    return train_dataset_ipt, train_dataset_opt, val_dataset_ipt, val_dataset_opt, test_dataset_ipt, test_dataset_opt, tokenize_ipt, tokenize_opt, vocab_ipt, vocab_opt


train_dataset_ipt, train_dataset_opt, val_dataset_ipt, val_dataset_opt, test_dataset_ipt, test_dataset_opt, tokenize_ipt, tokenize_opt, vocab_ipt, vocab_opt = load_dataset(opt)

def encode(x, vocab):
    return [vocab.stoi[s] for s in x]

def decode(x, vocab):
    return [vocab.itos[s] for s in x]


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def get_model(config, ipt_vocab_len, opt_vocab_len):
    model = build_transformer(ipt_vocab_len, opt_vocab_len, config['seq_len'], config['seq_len'])
    return model


ipt_vocab_len = len(vocab_ipt.stoi)
opt_vocab_len = len(vocab_opt.stoi)

def translate(config, sentence : str):
    seq_len = config['seq_len']

    sos_token_ipt = torch.tensor([vocab_ipt.stoi['<sos>']], dtype=torch.int64, device=device)
    eos_token_ipt = torch.tensor([vocab_ipt.stoi['<eos>']], dtype=torch.int64, device=device)
    pad_token_ipt = torch.tensor([vocab_ipt.stoi['<pad>']], dtype=torch.int64, device=device)

    sos_token_opt = torch.tensor([vocab_opt.stoi['<sos>']], dtype=torch.int64, device=device)
    eos_token_opt = torch.tensor([vocab_opt.stoi['<eos>']], dtype=torch.int64, device=device)
    pad_token_opt = torch.tensor([vocab_opt.stoi['<pad>']], dtype=torch.int64, device=device)

    model = get_model(config, ipt_vocab_len=ipt_vocab_len, opt_vocab_len=opt_vocab_len).to(device)
    state = torch.load('model.pt')
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    with torch.no_grad():
        source = tokenize_ipt.tokenizer(sentence)
        enc_num_padding_tokens = seq_len - len(source) - 2
        encoder_input = torch.cat(
            [
                sos_token_ipt,
                torch.tensor(encode(source, vocab_ipt), dtype=torch.int64, device=device),
                eos_token_ipt,
                torch.tensor([pad_token_ipt] * enc_num_padding_tokens, dtype=torch.int64, device=device)
            ],
            dim=0
        ).unsqueeze(0).to(device) # --> (1, seq_len)
        encoder_mask = (encoder_input != sos_token_ipt).long().to(device)
        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = []
        while len(decoder_output) < seq_len:
            decoder_input = torch.cat([
                sos_token_opt,
                torch.tensor(decoder_output, dtype=torch.int64, device=device),
                torch.tensor([pad_token_opt] * (seq_len - len(decoder_output) - 1), dtype=torch.int64, device=device),
            ], dim=0).unsqueeze(0).to(device)
            decoder_mask = (decoder_input != pad_token_opt).long().to(device)
            out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask).to(device)
            prob = model.project(out)
            prob = prob[0][len(decoder_output)]
            _, next_word = torch.max(prob, dim=0)
            if next_word.item() == vocab_opt.stoi['<eos>']:
                break

            decoder_output.append(next_word.item())

    return " ".join(vocab_opt.itos[x] for x in decoder_output)

print(translate(opt, "toi muon tro thanh mot AI researcher noi tieng tren the gioi"))

# Thụy Điển đã cố gắng đóng một vai trò tích cực hơn

# Các buổi trình diễn âm nhạc


# Năm 2013, Hòa Hiệp góp mặt ở trong chương trình truyền hình đình đám Bước nhảy hoàn vũ.