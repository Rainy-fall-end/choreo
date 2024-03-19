import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 64
max_len = 24
d_model = 1024
n_layers = 6
n_heads = 16
ffn_hidden = 2048
drop_prob = 0.1

# optimizer parameter setting
init_lr = 1e-6
factor = 0.9
adam_eps = 5e-8
patience = 10
warmup = 100
epoch = 100
clip = 1.0
weight_decay = 5e-6
inf = float('inf')
