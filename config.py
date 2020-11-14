import torch
# Encoder
hidden_size = 512
voc_size = 10000
emb_dim = 256
enc_n_layer = 4
dec_n_layer = 1
batch_size = 32
max_oov = 5
DEVICE = 'cuda:4' if torch.cuda.is_available() else 'cpu'
dropout = 0.1
