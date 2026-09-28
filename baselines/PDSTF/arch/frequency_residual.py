import torch
import torch.nn as nn

from .stfe import STFE


class FrequencyResidualLearning(nn.Module):

    def __init__(self, num_nodes, seq_len, pred_len, embed_size, hidden_size, fc_hidden_size):
        super().__init__()
        self.input_embedding = nn.Parameter(torch.randn(1, embed_size))
        self.encoder = STFE(num_nodes, seq_len, embed_size, hidden_size)
        self.prediction_head = nn.Sequential(
            nn.Linear(seq_len * embed_size, fc_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_size, pred_len)
        )

    def embed_input(self, x):
        x = x.unsqueeze(3)
        return x * self.input_embedding

    def forward(self, R_in):
        batch_size, _, num_nodes = R_in.shape
        x = self.embed_input(R_in)
        x = self.encoder(x)
        R_out = self.prediction_head(
            x.transpose(1, 2).reshape(batch_size, num_nodes, -1)
        ).permute(0, 2, 1)
        return R_out
