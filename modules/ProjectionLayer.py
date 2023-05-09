import hashlib
import torch
import torch.nn as nn

MAX_HASH_VALUE = 2 ** 8

class ProjectionLayer(nn.Module):
    def __init__(self, max_len, embedding_dim, hidden_dim, *args, **kwargs):
        # max_len = 64, embedding_dim = 420, hidden_dim=768
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        super(ProjectionLayer, self).__init__(*args, **kwargs)
        self.linear = nn.Linear(embedding_dim, hidden_dim)

        # self.hash_fn1 = lambda data: int.from_bytes(hashlib.new('sha256').digest(), 'little')
        # self.hash_fn2 = lambda data: int.from_bytes(hashlib.new('sha224').digest(), 'little')

        self.hash_fn1 = lambda data: int.from_bytes(hashlib.sha256(str(data).encode()).digest(), 'little')
        self.hash_fn2 = lambda data: int.from_bytes(hashlib.sha224(str(data).encode()).digest(), 'little')
    def forward(self, x):
        # print(f"x{x}")
        features = x["tokens"]['input_ids']
        # print(f"features{features}")
        #print(features.shape) #10,64

        # print(f"features.shape[0]{features.shape[0]}")
        # print(f"self.hash_fn1(features[0][0]){self.hash_fn1(features[0][0])}")
        # print(f"self.hash_fn1(features[0][1]){self.hash_fn1(features[0][1])}")

        hash1 = [[self.hash_fn1(features[i][j]) for j in range(self.max_len)]
                 for i in range(features.shape[0])]

        # print(f"hash1{hash1}")
        hash2 = [[self.hash_fn2(features[i][j]) for j in range(self.max_len)]
                 for i in range(features.shape[0])]

        # print(f"hash2{hash2}")
        hash3 = torch.Tensor([[[(hash1[i][j] + k * hash2[i][j]) % MAX_HASH_VALUE
                                for k in range(self.embedding_dim)]
                               for j in range(self.max_len)]
                             for i in range(features.shape[0])]).cuda()

        # print(f"hash3{hash3}")
        out = self.linear(hash3)
        # print(f"projectout.shape is{out.shape}")256 64 768
        return out

