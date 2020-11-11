import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
import config

class Encoder(nn.Module):
    def __init__(self):
        self.embedding = nn.Embedding(config.voc_size, config.emb_dim)
        self.gru = nn.GRU(config.emb_dim, config.hidden_size,
                          num_layers=config.encoder_n_layers, bidirectional=True, batch_first=True)
        # self.w_e = nn.Linear(2*config.hidden_size, 2*config.hidden_size)

    def forward(self, encoder_input):
        '''
        :param encoder_input: B, T
        :return: enc_out: B, T, 2H   enc_hidden: 2L, B, T
        '''
        embedded = self.embedding(encoder_input)  # B,T,E
        enc_out, enc_hidden = self.gru(embedded)  # enc_out: B, T, 2H   enc_hidden: 2L, B, T
        return enc_out, enc_hidden


class Decoder(nn.Module):
    def __init__(self):
        self.embedding = nn.Embedding(config.voc_size, config.emb_dim)
        self.gru = nn.GRU(config.emb_dim + 2*config.hidden_size, config.hidden_size, num_layers=1, batch_first=True)
        self.Wi = nn.Linear(2*config.hidden_size, config.hidden_size) # initial only(inherited from last encoder state)
        self.Wg = nn.Linear(config.hidden_size, config.voc_size) # hidden -> score_g
        self.Wc = nn.Linear(config.hidden_size*2, config.hidden_size)  # enc_out -> score_c

    def forward(self, decoder_input, encoder_outputs, encoder_idxs, prev_state, selective_read, step):
        '''
        :param decoder_input: B, 1
        :param encoder_outputs:  B,T,2H
        :param encoder_idxs: B,T
        :param prev_state: 1, B, H
        :param selective_read: B, 1, 2H
        :return: prob_out: B, 1, V'   decode_hidden: B, H    selective_read: B, 1, 2H
        '''
        # 0. set initial state
        if step == 0:
            prev_state = self.Wi(encoder_outputs[:, -1]).unsqueeze(0)  # prev_state: 1,B,H
            selective_read = torch.Tensor(config.batch_size, 1, 2*config.hidden_size).zero_()  # B, 1, 2H

        # 1. update state
        embedded = self.embedding(decoder_input)  # B, 1, E
        gru_input = torch.cat([selective_read, embedded], 2) # gru_input: B, 1, 2H+E
        _, decode_hidden = self.gru(gru_input, prev_state) # decoder_hidden: 1, B, H
        decode_hidden = decode_hidden.squeeze(0)  # B, H

        # 2. predict next word
        # 2.1 generate_mode
        score_g = self.Wg(decode_hidden)  # score_g: B, V

        # 2.2 copy_mode
        score_c = F.tanh(self.Wc((encoder_outputs).contiguous().view(-1, config.hidden_size*2)))  # BT, H
        score_c = score_c.view(config.batch_size, -1, config.hidden_size)  # B, T, H
        score_c = score_c.bmm(decode_hidden.unsqueeze(2)).squeeze() # (B,T,H)x(B,H,1) --> squeeze --> B,T
        score_c = F.tanh(score_c)

        encoder_mask = torch.Tensor(np.array(encoder_idxs == 0, dtype=float) * (-10000))  # B, T
        score_c = score_c + encoder_mask  # score_c: B, T

        # 2.3 combine copy and generate
        score_cat = torch.cat([score_g, score_c], 1) # B, T+V
        probs_cat = torch.softmax(score_cat, 1)  # B, T+V
        prob_g, prob_c = probs_cat[:config.voc_size], probs_cat[config.voc_size:]  # prob_g: B, V     prob_c: B, T

        # 2.4 expand vocab with unk
        oovs = torch.Tensor(config.batch_size, config.max_oov).zero_() + 1e-4  # B, max_oov
        prob_g = torch.cat([prob_g, oovs], 1) # B, V'

        # 2.5 add corresponding prob_c to prob_g
        encoder_idxs = torch.LongTensor(encoder_idxs).unsqueeze_(2) # B, T, 1
        one_hot = torch.FloatTensor(config.batch_size, encoder_outputs.size(1), prob_g.size(-1)).zero_() # one_hot: B, T, V'
        one_hot.scatter_(2, encoder_idxs, 1)
        prob_c_to_g = prob_c.unsqueeze_(1).bmm(one_hot).squeeze()   # B, V'
        prob_out = prob_c_to_g + prob_g  # B, V'
        prob_out.unsqueeze_(1)  # B, 1, V'

        # 3. get new selective read
        # 3.1 get tensor whether decoder input has previously appeared in Encoder
        id_from_encoder = []
        for b_id, each_batch_idx in enumerate(encoder_idxs):
            id_from_encoder.append([decoder_input[b_id] == idx for idx in each_batch_idx])
        # id_from_enc: B, T
        id_from_encoder = torch.Tensor(id_from_encoder)
        for b_id in range(config.batch_size):
            if id_from_encoder[b_id].sum().data[0] > 1:
                id_from_encoder[b_id] /= id_from_encoder[b_id].sum().data[0]
        # 3.2 weighted resprestation
        prob_appeared = prob_c * id_from_encoder  # prob_appeared: B, T
        prob_appeared.unsqueeze_(1)  # B, T, 1
        selective_read = prob_appeared.bmm(encoder_outputs) # B, 1, 2H
        return prob_out, decode_hidden, selective_read










