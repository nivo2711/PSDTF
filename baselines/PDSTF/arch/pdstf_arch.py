from argparse import Namespace

import torch
import torch.nn as nn

from .fdc_loss import frequency_domain_complementarity_loss
from .frequency_residual import FrequencyResidualLearning
from .periodic_pattern import PeriodicPatternLearning


class PDSTF(nn.Module):

    def __init__(self, **model_args):
        super().__init__()
        configs = Namespace(**model_args)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_nodes = configs.num_nodes
        self.daily_init_path = configs.daily_init_path
        self.weekly_init_path = configs.weekly_init_path
        self.adj = configs.adj
        self.fdc_weight = configs.fdc_weight
        self.low_frequency_bins = configs.low_frequency_bins
        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.fc_hidden_size = configs.fc_hidden_size
        self.daily_len = configs.time_of_day_size
        self.weekly_len = configs.day_of_week_size * self.daily_len

        self.periodic_branch = PeriodicPatternLearning(
            daily_len=self.daily_len,
            weekly_len=self.weekly_len,
            num_nodes=self.num_nodes,
            adj=self.adj,
            daily_init_path=self.daily_init_path,
            weekly_init_path=self.weekly_init_path,
        )
        self.residual_branch = FrequencyResidualLearning(
            self.num_nodes, self.seq_len, self.pred_len, self.embed_size,
            self.hidden_size, self.fc_hidden_size,
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs):
        x = history_data[..., 0]

        index_daily = history_data[..., 1] * self.daily_len
        index_daily = index_daily[:, -1, 0]
        index_weekly = (
            history_data[..., 1] * self.daily_len
            + history_data[..., 2] * self.weekly_len
        )
        index_weekly = index_weekly[:, -1, 0]

        S_in = self.periodic_branch(index_daily, index_weekly, self.seq_len)
        R_in = x - S_in
        R_out = self.residual_branch(R_in)

        S_out = self.periodic_branch(
            (index_daily + self.seq_len) % self.daily_len,
            (index_weekly + self.seq_len) % self.weekly_len,
            self.pred_len,
        )
        y = R_out + S_out

        loss_low, loss_high = frequency_domain_complementarity_loss(
            y, S_out, R_out, self.low_frequency_bins,
        )
        fdc_loss = (loss_low + loss_high) * self.fdc_weight
        return {"prediction": y.unsqueeze(-1), "fdc_loss": fdc_loss}
