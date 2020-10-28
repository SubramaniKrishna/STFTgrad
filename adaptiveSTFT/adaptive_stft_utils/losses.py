import torch.autograd
import torch
import torch.nn.functional as F
from .operators import clip_tensor_norm


def kurtosis(rfft_magnitudes_sq):
    epsilon = 1e-7
    max_norm = 0.1
    kur_part = [
        torch.sum(torch.pow(a, 2)) /
        (torch.pow(torch.sum(a), 2).unsqueeze(-1) + epsilon)
        for a in rfft_magnitudes_sq
    ]
    n_wnd = len(rfft_magnitudes_sq)
    assert n_wnd > 0
    catted = torch.cat(clip_tensor_norm(kur_part, max_norm=max_norm, norm_type=2)) / max_norm
    kur = torch.sum(catted) / n_wnd
    return kur

