import torch


def data_transformation_4_xformer(history_data: torch.Tensor, future_data: torch.Tensor, start_token_len: int):

    x_enc = history_data[..., 0]
    x_mark_enc = history_data[:, :, 0, 1:] - 0.5

    if start_token_len == 0:
        x_dec = torch.zeros_like(future_data[..., 0])
        x_mark_dec = future_data[..., :, 0, 1:] - 0.5
        return x_enc, x_mark_enc, x_dec, x_mark_dec
    else:
        x_dec_token = x_enc[:, -start_token_len:, :]
        x_dec_zeros = torch.zeros_like(future_data[..., 0])
        x_dec = torch.cat([x_dec_token, x_dec_zeros], dim=1)
        x_mark_dec_token = x_mark_enc[:, -start_token_len:, :]
        x_mark_dec_future = future_data[..., :, 0, 1:] - 0.5
        x_mark_dec = torch.cat([x_mark_dec_token, x_mark_dec_future], dim=1)

    return x_enc.float(), x_mark_enc.float(), x_dec.float(), x_mark_dec.float()
