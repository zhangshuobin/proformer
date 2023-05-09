import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads, dropout_rate):  # 420， 768， 8， 0.4
        super(Encoder, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, hidden_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, mask=None):
        # x: (seq_len, batch_size, input_dim)
        # mask: (batch_size, seq_len)
        # print(x.shape)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + attn_output
        x = self.norm1(x)
        # print(f"x.shape{x.shape}")
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self,  embedding_dim, hidden_dim, num_heads, n_layers, dropout_rate, *args, **kwargs):
        super(EncoderLayer, self).__init__()
        self.layers = nn.ModuleList([
            Encoder(embedding_dim, hidden_dim, num_heads, dropout_rate)
            for _ in range(n_layers)  # 2层encoder
        ])
        # self.avg_max_pooling = nn.MaxPool1d(3, 1, 1)局部最大池化
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # x: (seq_len, batch_size, input_dim)
        # mask: (batch_size, seq_len)
        for layer in self.layers:
            x = layer(x, mask)
        # x = x.mean(dim=1)平均池化
        # x = x.max(dim=1)[0]全局最大池化
        x = self.softmax(x.mean(dim=1))
        # print(f"encodex.shape is {x.shape}")
        return x
        # 10,64,768


