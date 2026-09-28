import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttention


class STAE(nn.Module):

    def __init__(self, period_len, num_nodes, adj):
        super().__init__()
        self.period_len = period_len
        self.num_nodes = num_nodes
        self.adj = adj

        self.graph_projection = nn.Linear(num_nodes, num_nodes)
        self.fusion_projection = nn.Linear(2 * num_nodes, num_nodes)
        self.temporal_attention = SelfAttention(num_nodes)
        self.spatial_attention = SelfAttention(period_len)

    def forward(self, embedding):
        data = torch.einsum('ln,nv->lv', (embedding, self.adj.to(embedding.device)))
        data = self.graph_projection(data)
        data = F.relu(data)

        temporal = self.temporal_attention(data)
        spatial = self.spatial_attention(data.transpose(0, 1)).transpose(0, 1)
        return self.fusion_projection(torch.cat([temporal, spatial], dim=-1))
