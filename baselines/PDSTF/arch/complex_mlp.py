import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexMLP(nn.Module):

    def __init__(self, embed_size, hidden_size, scale=0.02, sparsity_threshold=0.01):
        super().__init__()
        self.scale = scale
        self.sparsity_threshold = sparsity_threshold

        self.real_weight1 = nn.Parameter(scale * torch.randn(embed_size, hidden_size))
        self.imag_weight1 = nn.Parameter(scale * torch.randn(embed_size, hidden_size))
        self.real_bias1 = nn.Parameter(scale * torch.randn(hidden_size))
        self.imag_bias1 = nn.Parameter(scale * torch.randn(hidden_size))
        self.real_weight2 = nn.Parameter(scale * torch.randn(hidden_size, embed_size))
        self.imag_weight2 = nn.Parameter(scale * torch.randn(hidden_size, embed_size))
        self.real_bias2 = nn.Parameter(scale * torch.randn(embed_size))
        self.imag_bias2 = nn.Parameter(scale * torch.randn(embed_size))

    def forward(self, x):
        hidden_real = F.relu(
            torch.einsum('bijd,df->bijf', x.real, self.real_weight1)
            - torch.einsum('bijd,df->bijf', x.imag, self.imag_weight1)
            + self.real_bias1
        )
        hidden_imag = F.relu(
            torch.einsum('bijd,df->bijf', x.imag, self.real_weight1)
            + torch.einsum('bijd,df->bijf', x.real, self.imag_weight1)
            + self.imag_bias1
        )

        output_real = F.relu(
            torch.einsum('bijf,fd->bijd', hidden_real, self.real_weight2)
            - torch.einsum('bijf,fd->bijd', hidden_imag, self.imag_weight2)
            + self.real_bias2
        )
        output_imag = F.relu(
            torch.einsum('bijf,fd->bijd', hidden_imag, self.real_weight2)
            + torch.einsum('bijf,fd->bijd', hidden_real, self.imag_weight2)
            + self.imag_bias2
        )

        output = torch.stack([output_real, output_imag], dim=-1)
        output = F.softshrink(output, lambd=self.sparsity_threshold)
        return torch.view_as_complex(output)
