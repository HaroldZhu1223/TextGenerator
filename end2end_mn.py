import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import torch.nn.init as I

class MemN2N(nn.Module):
    def __init__(self, layer_wise):
        self.voc_size = config.voc_size
        self.emb_dim = config.emb_dim
        self.cand_mem_num = config.mem_num
        self.n_hops = config.n_layer
        self.use_lw = layer_wise

        self.Wa_list = nn.ModuleList([nn.Embedding(self.voc_size, self.emb_dim)])
        self.Wc_list = nn.ModuleList([nn.Embedding(self.voc_size, self.emb_dim)])
        self.Wa_list[-1].weight.data.normal_(0,0.1)
        self.Wc_list[-1].weight.data.normal_(0,0.1)
        if self.use_lw:
            for _ in range(1, self.n_hops):
                self.Wa_list.append(self.Wa_list[-1])
                self.Wc_list.append(self.Wc_list[-1])
            self.Wb = nn.Embedding(self.voc_size, self.emb_dim)
            self.Wb.weight.data.normal_(0,0.1)
            self.weight_out = nn.Parameter(
                I.normal_(torch.empty(size=(self.voc_size, self.emb_dim)), 0, 0.1)
            )
            self.H = nn.Linear(self.emb_dim, self.emb_dim)
            self.H.weight.data.normal_(0, 0.1)

        else:
            for _ in range(1, self.n_hops):
                self.Wa_list.append(self.Wc_list[-1])
                self.Wc_list.append(nn.Embedding(self.voc_size, self.emb_dim))
                self.Wc_list[-1].weight.data.normal_(0, 0.1)
            self.Wb = self.Wa_list[0]
            self.weight_out = self.Wc_list[-1].weight   # self.weighted_out: V, E

        self.TA = nn.Parameter(I.normal_(torch.empty(self.cand_mem_num, self.emb_dim)), 0, 0.1)
        self.TC = nn.Parameter(I.normal_(torch.empty(self.cand_mem_num, self.emb_dim)), 0, 0.1)

    def compute_weights(self, len):
        pass

    def forward(self, query, story):
        '''
        :param query:  1, Tq [tensor]
        :param story:  N, Tm [tensor]
        :return:
        '''
        len_q = query.size(-1)
        N, len_m = story.size()
        weights_q = self.compute_weights(len_q)   # Tq, E
        u = (self.Wb(query) * weights_q).sum(1)   # 1, E
        weights_m = self.compute_weights(len_m)   # Tm, E

        for i in range(self.n_hops):
            memory_in = (self.Wa_list[i](story.view(-1, len_m)) * weights_m).sum(1).view(*story.size()[:-1], -1) # N, E
            memory_in += self.TA  # N, E
            memory_out = (self.Wc_list[i](story.view(-1, len_m)) * weights_m).sum(1).view(*story.size()[:-1], -1)  # N, E
            memory_out += self.TC # N, E
            attn_weight = torch.sum(memory_in * u, dim = 1) # N
            attn_prob = F.softmax(attn_weight, dim=-1) # N
            weighted_out = torch.sum(memory_out.t() * attn_prob, dim=1).unsqueeze(0)  # 1, E
            if self.use_lw:
                u = u + self.H(weighted_out)
            else:
                u = u + weighted_out

        return F.log_softmax(F.linear(u, self.weight_out), dim = -1)  # F.linear(u, self.out): 1, V


class GMemN2N(nn.Module):
    def __init__(self, layer_wise):
        self.voc_size = config.voc_size
        self.emb_dim = config.emb_dim
        self.cand_mem_num = config.mem_num
        self.n_hops = config.n_layer
        self.use_lw = layer_wise

        self.Wa_list = nn.ModuleList([nn.Embedding(self.voc_size, self.emb_dim)])
        self.Wc_list = nn.ModuleList([nn.Embedding(self.voc_size, self.emb_dim)])
        self.Wt_list = nn.ModuleList([nn.Linear(self.emb_dim, self.emb_dim)])
        self.Wa_list[-1].weight.data.normal_(0,0.1)
        self.Wc_list[-1].weight.data.normal_(0,0.1)
        self.Wt_list[-1].weight.data.normal_(0,0.1)

        if self.use_lw:
            for _ in range(1, self.n_hops):
                self.Wa_list.append(self.Wa_list[-1])
                self.Wc_list.append(self.Wc_list[-1])
                self.Wt_list.append(self.Wt_list[-1])
            self.Wb = nn.Embedding(self.voc_size, self.emb_dim)
            self.Wb.weight.data.normal_(0,0.1)
            self.weight_out = nn.Parameter(
                I.normal_(torch.empty(size=(self.voc_size, self.emb_dim)), 0, 0.1)
            )
            self.H = nn.Linear(self.emb_dim, self.emb_dim)
            self.H.weight.data.normal_(0, 0.1)

        else:
            for _ in range(1, self.n_hops):
                self.Wa_list.append(self.Wc_list[-1])
                self.Wc_list.append(nn.Embedding(self.voc_size, self.emb_dim))
                self.Wc_list[-1].weight.data.normal_(0, 0.1)
                self.Wt_list.append(nn.Linear(self.emb_dim, self.emb_dim))
                self.Wt_list[-1].weight.data.normal_(0, 0.1)

            self.Wb = self.Wa_list[0]
            self.weight_out = self.Wc_list[-1].weight   # self.weighted_out: V, E

        self.TA = nn.Parameter(I.normal_(torch.empty(self.cand_mem_num, self.emb_dim)), 0, 0.1)
        self.TC = nn.Parameter(I.normal_(torch.empty(self.cand_mem_num, self.emb_dim)), 0, 0.1)

    def compute_weights(self, len):
        pass

    def forward(self, query, story):
        '''
        :param query:  1, Tq [tensor]
        :param story:  N, Tm [tensor]
        :return: 1, V
        '''
        len_q = query.size(-1)
        N, len_m = story.size()
        weights_q = self.compute_weights(len_q)   # Tq, E
        u = (self.Wb(query) * weights_q).sum(1)   # 1, E
        weights_m = self.compute_weights(len_m)   # Tm, E

        for i in range(self.n_hops):
            memory_in = (self.Wa_list[i](story.view(-1, len_m)) * weights_m).sum(1).view(*story.size()[:-1], -1) # N, E
            memory_in += self.TA  # N, E
            memory_out = (self.Wc_list[i](story.view(-1, len_m)) * weights_m).sum(1).view(*story.size()[:-1], -1)  # N, E
            memory_out += self.TC # N, E
            attn_weight = u @ memory_in.view(-1, self.emb_dim) # 1, N
            attn_prob = F.softmax(attn_weight, dim=-1) # 1, N
            weighted_out = attn_prob @ memory_out  # 1, E
            t = F.sigmoid(self.Wt_list[i](u)) # 1, E
            if self.use_lw:
                u = u*(1-t) + self.H(weighted_out)*t  # 1, E
            else:
                u = u*(1-t) + weighted_out*t

        return F.log_softmax(F.linear(u, self.weight_out), dim = -1)  # F.linear(u, self.out): 1, V