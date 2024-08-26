import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, num_channels):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_channels = num_channels
        self.linear = nn.Linear(patch_len * num_channels, d_model)

    def forward(self, x):
        # x shape: [batch_size, num_channels, sequence_length]
        batch_size, num_channels, sequence_length = x.shape
        
        # Reshape and transpose to [batch_size, sequence_length, num_channels]
        x = x.permute(0, 2, 1)
        
        # Create patches
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches shape: [batch_size, num_patches, patch_len, num_channels]
        
        # Reshape patches to [batch_size, num_patches, patch_len * num_channels]
        patches = patches.reshape(batch_size, -1, self.patch_len * self.num_channels)
        
        # Apply linear projection
        return self.linear(patches)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.d_k)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k).float())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_length, self.d_model)
        return self.out_proj(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class PatchTSTEncoder(nn.Module):
    def __init__(self, num_channels, patch_len, stride, d_model, num_heads, num_layers, d_ff, max_len=5000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, num_channels)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x):
        # x shape: [batch_size, num_channels, sequence_length]
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x