import torch
import torch.nn.functional as F


def frequency_domain_complementarity_loss(y, S_out, R_out, low_frequency_bins):
    y_fft = torch.fft.rfft(y, dim=1)

    low_fft = torch.zeros_like(y_fft)
    low_fft[:, :low_frequency_bins] = y_fft[:, :low_frequency_bins]

    high_fft = torch.zeros_like(y_fft)
    high_fft[:, low_frequency_bins:] = y_fft[:, low_frequency_bins:]

    low_time = torch.fft.irfft(low_fft, n=y.size(1), dim=1)
    high_time = torch.fft.irfft(high_fft, n=y.size(1), dim=1)

    loss_low = F.mse_loss(S_out, low_time)
    loss_high = F.mse_loss(R_out, high_time)
    return loss_low, loss_high
