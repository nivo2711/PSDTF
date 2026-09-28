import numpy as np
import torch
import torch.nn as nn

from .stae import STAE


class PeriodicEmbedding(nn.Module):

    def __init__(self, init_path, period_len, num_nodes):
        super().__init__()
        init_data = np.load(init_path)
        assert init_data.shape == (period_len, num_nodes), \
            f"Expected shape {(period_len, num_nodes)}, but got {init_data.shape}"
        self.weight = nn.Parameter(torch.from_numpy(init_data).float(), requires_grad=True)

    def forward(self):
        return self.weight


class PeriodicPatternLearning(nn.Module):

    def __init__(self, daily_len, weekly_len, num_nodes, adj,
                 daily_init_path, weekly_init_path):
        super().__init__()
        self.daily_len = daily_len
        self.weekly_len = weekly_len
        self.num_nodes = num_nodes

        daily_encoder = STAE(daily_len, num_nodes, adj)
        self.daily_embedding = PeriodicEmbedding(daily_init_path, daily_len, num_nodes)
        self.daily_encoder = daily_encoder
        weekly_encoder = STAE(weekly_len, num_nodes, adj)
        self.weekly_embedding = PeriodicEmbedding(weekly_init_path, weekly_len, num_nodes)
        self.weekly_encoder = weekly_encoder

    def forward(self, index_daily, index_weekly, length):
        daily_index = (
            index_daily.view(-1, 1)
            + torch.arange(length, device=index_daily.device).view(1, -1)
        ) % self.daily_len
        daily_pattern = self.daily_encoder(self.daily_embedding())
        daily_pattern = daily_pattern[daily_index.long()]

        weekly_index = (
            index_weekly.view(-1, 1)
            + torch.arange(length, device=index_weekly.device).view(1, -1)
        ) % self.weekly_len
        weekly_pattern = self.weekly_encoder(self.weekly_embedding())
        weekly_pattern = weekly_pattern[weekly_index.long()]
        return daily_pattern + weekly_pattern
