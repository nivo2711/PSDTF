import torch
import torch.nn as nn

from .complex_mlp import ComplexMLP


class STFE(nn.Module):

    def __init__(self, num_nodes, seq_len, embed_size, hidden_size,
                 scale=0.02, sparsity_threshold=0.01):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.spatial_mlp = ComplexMLP(embed_size, hidden_size, scale, sparsity_threshold)
        self.temporal_mlp = ComplexMLP(embed_size, hidden_size, scale, sparsity_threshold)

    def spatial_frequency_learning(self, x):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.spatial_mlp(x)
        return torch.fft.irfft(y, n=self.num_nodes, dim=2, norm='ortho')

    def temporal_frequency_learning(self, x):
        x = x.transpose(1, 2)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.temporal_mlp(x)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm='ortho')
        return x.transpose(1, 2)

    def forward(self, x):
        skip = x
        x = self.spatial_frequency_learning(x)
        x = self.temporal_frequency_learning(x)
        return x + skip
