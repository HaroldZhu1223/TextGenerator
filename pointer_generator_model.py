import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import config




def init_linear_wt(linear):
    linear.weight.data.normal_(std = config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std = config.trunc_norm_init_std)

def init_wt_normal(weights):
    weights.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(weights):
    weights.data.uniform_(-config.unif, config.unif)

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.voc_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_size, config.encoder_n_layers, batch_first=True, bidirectional=True)
        self.W_h = nn.Linear(config.hidden_size*2, config.hidden_size*2)

    def forward(self, encoder_input, seq_len):
        '''
        :param encoder_input: B, T
        :param seq_len: 1
        :return: encoder_outputs: B, T, 2*H   encoder_feature: B*T, 2*H    hidden: L*2, B, H
        '''
        embedded = self.embedding(encoder_input) # embedded: B, T, emb_dim
        packed = nn.utils.rnn.pack_padded_sequence(embedded, seq_len, batch_first=True)
        encoder_outputs, hidden = self.lstm(packed) # hidden: （hn, cn）   hn: 2L,B,H    cn:2L,B,H
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs)  # encoder_outputs: B, T, hidden_size*2
        encoder_feature = encoder_outputs.contiguous().view(-1, 2*config.hidden_size) # encoder_feature: B*T, 2*hidden,size
        encoder_feature = self.W_h(encoder_feature)
        return encoder_outputs, encoder_feature, hidden

class RuduceState(nn.Module):
    def __init__(self):
        super(RuduceState, self).__init__()
        self.reduce_h = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.reduce_c = nn.Linear(2*config.hidden_size, config.hidden_size)
        init_linear_wt(self.reduce_h)
        init_linear_wt(self.reduce_c)

    def forward(self, encoder_hidden):
        '''
        :param encoder_hidden: hn, cn       2,B,H * 2
        :return:
        '''
        h, c = encoder_hidden
        h_in = h.transpose(0,1).contiguous().view(-1, 2*config.hidden_size) # h.t: B,2L,H ->  h_in: BL,2H
        hidden_reduce_h = F.relu(self.reduce_h(h_in))  # h_in: BL, H
        c_in = c.transpose(0, 1).contiguous().view(-1, 2 * config.hidden_size)  # c.t: B,2L,H ->  c_in: BL,2H
        hidden_reduce_c = F.relu(self.reduce_c(c_in))  # c_in: BL, H
        return (hidden_reduce_h.unsqueeze(0), hidden_reduce_c.unsqueeze(0)) # ( (1, BL, H), (1, BL, H) )



class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        if config.is_coverge:
            self.w_c = nn.Linear(1, config.hidden_size * 2, bias=False)
        self.w_d = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.v = nn.Linear(config.hidden_size*2,1)

    def forward(self, decoder_input, encoder_outputs, encoder_feature, encoder_padding_mask, coverage):
        '''
        :param decoder_input: B, 2H
        :param encoder_outputs: B, T, 2H
        :param encoder_feature:  BT, 2H
        :param encoder_padding_mask: B*T
        :param coverage: B, T
        :return: context_vec: B, 2H    prob_masked: B, T     coverage: B, T
        '''
        B, T, N = list(encoder_outputs.size())
        dec_feature = self.w_d(decoder_input)  # attn_score: B, 2H
        dec_feature = dec_feature.unsqueeze(1).expand(B, T, N).contiguous() # dec_attn_exp: B, T, 2H
        dec_feature = dec_feature.view(-1, N) # dec_attn_exp: B*T, 2H
        attn_feature = encoder_feature + dec_feature # attn_feature: B*T, 2H
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B*T, 1
            coverage_feature = self.w_c(coverage_input) # B*T, 2H
            attn_feature = attn_feature + coverage_feature
        score = self.v(F.tanh(attn_feature))  #socre: B*T, 1
        score = score.view(-1, T) # score: B, T

        prob_masked = F.softmax(score, dim=1) * encoder_padding_mask  # prob_masked: B, T
        norm_term = torch.sum(prob_masked, dim=1)
        prob_masked = (prob_masked / norm_term).unsqueeze_(1)   # prob_masked: B,1,T
        # encoder_outputs: B, T, 2H
        context_vec = prob_masked.bmm(encoder_outputs).view((-1, config.hidden_size*2)) # B,1, 2H -> B, 2H

        prob_masked = prob_masked.squeeze(1) # B, T

        if config.is_coverage:
            coverage = coverage.view(-1, T) # B, T
            coverage = coverage + prob_masked # B, T

        return context_vec, prob_masked, coverage


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attn = Attention()
        self.embedding = nn.Embedding(config.voc_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        self.w_d_context = nn.Linear(config.hidden_size * 2 + config.emb_dim, config.hidden_size)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_size*4+config.emb_dim, 1)
        self.out1 = nn.Linear(config.hidden_size*3, config.hidden_size)
        self.out2 = nn.Linear(config.hidden_size*3, config.voc_size)
        init_linear_wt(self.out2)

    def forward(self, decoder_input, prev_hidden, encoder_outputs, encoder_feature, enc_pad_mask, context_vec, coverage, step, extra_zeros, enc_batch_extend_vocab):
        '''
        :param decoder_input: B, 1
        :param prev_hidden: 1, B, H
        :param encoder_outputs: B, T, 2H
        :param encoder_feature: BT, 2H
        :param enc_pad_mask:
        :param coverage: B, T
        :return:
        '''
        if not self.training and step == 0:
            h_decoder, c_decoder = prev_hidden # 1,B,H
            cur_hidden = torch.cat((h_decoder.view(-1, config.hidden_size), c_decoder.view(-1, config.hidden_size)), 1) #  B, 2H
            c_t, _, coverage_next = self.attn(cur_hidden, encoder_outputs, encoder_feature, enc_pad_mask, coverage)
            coverage = coverage_next

        embedded = self.embedding(decoder_input) # B, 1, E
        x = self.w_d_context(torch.cat((context_vec, embedded), 1)) # B, 1, E
        lstm_outputs, hidden = self.lstm(x, prev_hidden)  # lstm_out: B, 1, H    hidden: 1, B, H

        h_decoder, c_decoder = hidden
        cur_hidden = torch.cat((h_decoder.view(-1, config.hidden_size), c_decoder.view(-1, config.hidden_size)), 1) # B, 2H
        context_vec, c_prob, coverage_next = self.attn(cur_hidden, encoder_outputs, encoder_feature, enc_pad_mask, coverage)

        if coverage or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((context_vec, cur_hidden, x), 1)   # B, 4*H+E
            p_gen = self.p_gen_linear(p_gen_input)   # B, 1
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_outputs.view(-1, config.hidden_size), ), 1)   # B, 3H
        output = self.out1(output)  # B, H
        out_v = self.out2(output)   # B, V
        v_prob = F.softmax(out_v, dim = 1) # B,V
        if config.pointer_gen:
            vocab_prob = p_gen * v_prob # B,V
            copy_prob = (1-p_gen) * c_prob # B, T
            if extra_zeros is not None:
                vocab_prob_extend = torch.cat([vocab_prob, extra_zeros], 1)

            total_prob = vocab_prob_extend.scatter_add_(1, enc_batch_extend_vocab, copy_prob) # B, V'
        else:
            total_prob = v_prob
        return total_prob, hidden, context_vec, c_prob, p_gen, coverage







