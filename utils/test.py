import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.func import clones


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(2,3)

    def forward(self, x):
        return self.emb(x) * math.sqrt(3)

if __name__ == '__main__':
    V = 100
    E = 32
    N = 20
    C = 16
    L = 10
    B = 2
    T= 3
    state = torch.ones(size = (1, 2))
    state2 = torch.ones(size=(1, 2))

    u = torch.ones(size = (N, 1))
    memory = torch.ones(size = (N,E))
    emb = Embedding()

