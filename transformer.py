import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from utils.func import clones
import numpy as np
import copy

def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), 1).astype('int')
    return torch.from_numpy(mask) == 0

class Batch:
    def __init__(self, src, trg = None, pad = 0):
        '''
        :param src: B, T
        :param trg: B, T
        :param pad:
        '''
        self.src = src
        self.pad = pad
        self.src_mask = (src != pad).unsqueeze(-2)   # 1, B, T
        if trg:
            self.trg_in = trg[:,:-1]  # B, T-1
            self.trg_out = trg[:,1:]  # B, T-1
            self.trg_mask = self.make_std_mask(self)   # 1, B, T-1
            self.real_length = (self.trg_out != self.pad).sum()  


    def mask_std_mask(self):
        trg_mask = (self.trg_in != self.pad).unsqueeze(-2) & subsequent_mask(self.trg_in.size(-1))  # 1, B,T-1
        return trg_mask




class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(config.voc_size, config.hidden_size)

    def forward(self, x):
        return self.emb(x) * math.sqrt(config.hidden_size)

class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        self.max_pos_len = 5000
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(p = config.pe_drop)
        self.pe = torch.empty(self.max_pos_len, self.hidden_size, device=config.DEVICE, requires_grad=False) # pe: max, H
        pos = torch.arange(0, self.max_pos_len).unsqueeze(1) # pos: max, 1
        div_term = torch.exp(2*torch.arange(0,self.max_pos_len, 2, device=config.DEVICE)/self.emb_dim * (-math.log(10000))) # div_term: H/2
        self.pe[:, 0::2] = pos * math.sin(div_term)
        self.pe[:, 1::2] = pos * math.cos(div_term)
        self.pe.unsqueeze_(0) # 1, max, H

    def forward(self, x):
        '''
        :param x: B, T, H
        :return: B, T, H
        '''
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)


def attention(q, k, v, mask = None, dropout = None):
    '''
    :param q: B,h,T,dk
    :param k:
    :param v: B,h,T,dk
    :param mask: B,T
    :param dropout:
    :return:
    '''
    dk = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(dk)  # q:B,h,T,dk   k_t:B,h,dk,T  --> scores: B,h,T,T
    if mask is not None:
        scores.masked_fill_(mask == 0, -1e8)
    attn_prob = F.softmax(scores, dim=-1)  # B, h, T, T
    if dropout:
        attn_prob = dropout(attn_prob)
    new_represent = torch.matmul(attn_prob, v) # B,h,T,dk
    return torch.matmul(attn_prob, v), attn_prob


class Mutihead_attn(nn.Module):
    def __init__(self):
        super(Mutihead_attn, self).__init__()
        self.hidden_size = config.hidden_size
        self.head_num = config.head_num
        self.dk = self.emb_dim / self.head_num
        self.attn_drop = config.attn_drop
        self.linears = clones(nn.Linear(config.hidden_size, config.hidden_size), 4)
        self.attn = None
        self.attn_drop = nn.Dropout(p=config.attn_drop)

    def forward(self, query, key, value, mask = None):
        '''
        :param query: B, T, H
        :param key:
        :param value:
        :param mask: B, T
        :return: B,T,H
        '''
        B = query.size(0)
        query, key, value = [l(x).view(B, -1, self.head_num, self.dk).transpose(1,2)
                             for l, x in zip(self.linears, (query, key, value))]  # query: B,h,T,dk
        new_represt, attn_prob = attention(query, key, value, mask = mask, dropout = self.attn_drop)  # new: B,h,T,dk  attn_prob: B,h,T,T
        x = new_represt.transpose(1,2).contiguous().view(B,-1,self.dk*self.head_num) # B,T,E
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zero(hidden_size))
        self.eps = eps

    def forward(self, x):
        '''
        :param x: B, T, H
        :return:
        '''
        mean = x.mean(-1, keep_dim = True)
        std = x.std(-1, keep_dim =True)
        return self.alpha * (x - mean) / torch.sqrt(std**2 + self.eps) + self.beta

class SubLayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        '''
        :param x: B, T, H
        :param sublayer:
        :return:
        '''
        return x + self.dropout(self.norm(sublayer(x)))

class PostionwiseFeedForward(nn.Module):
    def __init__(self, dropout = 0.1):
        super(PostionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(config.hidden_size, config.ff_dim)
        self.w_2 = nn.Linear(config.ff_dim, config.hidden_size)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))  # B,T,H

# Encoder
class Encoder(nn.Module):
    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.layer_connect = clones(SubLayerConnection(hidden_size, dropout), 2)
        self.hidden_size = hidden_size

    def forward(self, x, mask):
        x = self.layer_connect[0](x, lambda x: self.self_attn(x,x,x,mask))
        return self.layer_connect[1](x, self.feed_forward)


# Decoder
class Decoder(nn.Module):
    def __init__(self, layer, n):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, trg_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.trg_attn = trg_attn
        self.src_attn = src_attn
        self.ff = feed_forward
        self.sublayer = clones(SubLayerConnection(self.hidden_size, dropout),3)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        x = self.sublayer[0](x, lambda x:self.src_attn(x,x,x,trg_mask))
        x = self.sublayer[1](x, lambda x:self.src_attn(x, encoder_output, encoder_output, src_mask))
        return self.sublayer[2](x, self.ff)


# Trm
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_emb, trg_emb, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.trg_emb = trg_emb
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_emb(src), src_mask)

    def decode(self, trg, encode_output, src_mask, trg_mask):
        return self.decoder(self.trg_emb(trg), encode_output, src_mask, trg_mask)

    def forward(self, src, trg, src_mask, trg_mask):
        return self.decode(trg, self.encode(src, src_mask), src_mask, trg_mask)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Linear(config.hidden_size, config.voc_size)

    def forward(self, x):
        '''
        :param x: B, T, H
        :return:
        '''
        return F.softmax(self.linear(x), dim=-1)

def make_model():
    c = copy.deepcopy()
    attn = Mutihead_attn().to(config.DEVICE)
    ff = PostionwiseFeedForward().to(config.DEVICE)
    pos_enc = PositionalEncoding()

    trm = Transformer(
        Encoder(EncoderLayer(config.hidden_size, c(attn), c(ff), config.dropout).to(config.DEVICE), config.N).to(config.DEVICE),
        Decoder(DecoderLayer(config.hidden_size, c(attn), c(attn), c(ff), config.dropout).to(config.DEVICE), config.N).to(config.DEVICE),
        nn.Sequential(Embedding().to(config.DEVICE), c(pos_enc)),
        nn.Sequential(Embedding().to(config.DEVICE), c(pos_enc)),
        Generator().to(config.DEVICE)
    )
    for p in trm.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return trm.to(config.DEVICE)





