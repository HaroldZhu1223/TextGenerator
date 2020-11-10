import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Encoder
class Encoder(nn.Module):
    def __init__(self, config, dropout = 0.1):
        super(Encoder, self).__init__()
        # n_layer, hidden_size, voc_size, embedding_weights
        self.n_layers = config.n_layers
        self.hidden_size = config.hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.load(config.init_emb_path))
        self.gru = nn.GRU(config.embedding_dim, config.hidden_size, config.encoder_n_layers,
                          dropout = (0 if config.encoder_n_layers == 1 else dropout), bidirectional=True)

    def forward(self, inp_vec, seq_len, hidden = None):
        '''
        :param inp_vec:  padded  size: T,B
        :param seq_len:
        :param hidden:
        :return:
        '''
        embed = self.embedding(inp_vec)  # embed: T, B, emb_dim
        packed = nn.utils.rnn_pack_padded_sequence(embed, seq_len)  # packed: T,B,emb_dim 压紧
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn_pad_packed_sequence(outputs) # outputs: T,B,d*hid_size
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:] # outputs: T,B,hid_size
        return outputs, hidden # hidden: num_layers*d, batch,hid_size


class Attn(nn.Module):
    def __init__(self, method, config):
        super(Attn, self).__init__()
        if method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, 'is not an approriate attention method.')
        self.hidden_size = config.hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size*2, self.hidden_size)
            self.v = nn.Parameter(self.hidden_size)

    def dot_score(self, decoder_hidden, encoder_outputs):
        '''
        :param decoder_hidden:  1, B, h
        :param encoder_outputs:  T, B, h
        :return: T,B
        '''
        return torch.sum(encoder_outputs* decoder_hidden, dim=2)

    def general_score(self, decoder_hidden, encoder_outputs):
        score = self.attn(encoder_outputs)
        return torch.sum(score*decoder_hidden, dim = 2)

    def concat_score(self, decoder_hidden, encoder_outputs):
        concat = self.attn(torch.cat((decoder_hidden.expand(encoder_outputs.size(0), -1, -1), encoder_outputs), 2))
        return torch.sum(self.v * concat, dim=2)


    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == 'general':
            attn_score = self.general_score(decoder_hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_score = self.concat_score(decoder_hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_score = self.dot_score(decoder_hidden, encoder_outputs)
        # tranpose T,B -> B, T
        attn_score = attn_score.t()
        return F.softmax(attn_score, dim=1).unsqueeze(1) # B,1,T


class Decoder(nn.Module):
    def __init__(self, config, dropout = 0.1):
        self.hidden_size = config.hidden_size
        self.emb_dim = config.embedding_dim
        self.voc_size = config.voc_size
        self.n_layers = config.decoder_n_layers
        self.dropout = config.dropout
        self.embedding = nn.Embedding.from_pretrained(torch.load(config.init_emb_path))
        # self.embedding = nn.Embedding(self.voc_size, self.embed_dim)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(self.emb_dim, self.hidden_size, self.decoder_n_layers,
                          dropout = 0 if self.n_layers == 1 else dropout)
        self.concat = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.voc_size)
        self.attn = Attn(config.method, config)


    def forward(self, decoder_input, prev_hidden, encoder_outputs):
        '''
        :param decoder_input: 1(T), B
        :param prev_hidden: layer, H
        :param encoder_outputs: T, B, H
        :return: decoder_output: B,1    hidden: L, B, H
        '''
        embed = self.embedding(decoder_input) # 1, B, H
        embed = self.emb_dropout(embed)
        gru_output, hidden = self.gru(embed, prev_hidden) # hidden: layer, B, H     gru_output: 1, B, H
        attn_weights = self.attn(gru_output, encoder_outputs) # B, 1, T
        encoder_outputs_t = encoder_outputs.transpose(0,1) # B, T, H
        context_vec = attn_weights.bmm(encoder_outputs_t) # bmm B个(1,T)(T,H)相乘得到 --> context_vec: B, 1, H
        # concat_output -> concat  [gru_output;context_vec]
        # 统一维度
        gru_output = gru_output.squeeze(0)    # 1,B,H -> B, H
        context_vec = context_vec.squeeze(1)  # B, 1, H —> B, H
        concat_output = torch.cat((gru_output, context_vec), dim=1)  # B, 2*H
        concat_output = torch.tanh(self.concat(concat_output)) # B, H
        decoder_output = self.out(concat_output) # B, V
        decoder_output = torch.softmax(decoder_output, dim=1) # B, 1
        return decoder_output, hidden






