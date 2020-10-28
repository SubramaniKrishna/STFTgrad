import math
import sys
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.insert(0, pathlib.Path(__file__).parent.parent.parent.absolute())
from UMNN.models.UMNN import MonotonicNN


# Monotonically increasing mapping
class IdxToWindow(nn.Module):
    def __init__(self, signal_len, num_windows=80, baseline_mapping_trick=True):
        super(IdxToWindow, self).__init__()
        self.signal_len = signal_len
        self.num_windows = num_windows
        self.baseline_mapping_trick = baseline_mapping_trick
        self.slope = nn.Parameter(torch.tensor(
            signal_len / num_windows, dtype=torch.float32, requires_grad=True, device="cuda"))
        self.overlap_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.bias = nn.Parameter(torch.tensor(
            1e-2, dtype=torch.float32, requires_grad=True, device="cuda"))
        self.model_monotonic = MonotonicNN(
            2, [128, 128, 128], nb_steps=768, dev="cuda").cuda()
        self.signal_mid_point = nn.Parameter(torch.tensor(
            signal_len / 2.0, dtype=torch.float32, requires_grad=True, device="cuda"))

    def forward(self, idx):
        assert len(idx.shape) == 1
        # transform window idx
        # scale down by 10
        rescale = .1
        in_var = (idx * rescale + self.bias).view(idx.size(0), 1)
        stem = (self.model_monotonic(in_var, in_var).flatten() / rescale)
        # at least advance by 32 sample per window
        if self.baseline_mapping_trick:
            baseline_mapping = 32 * idx
        else:
            baseline_mapping = 0
        # convert window idx to sample idx
        x_i = (stem * self.slope + baseline_mapping) + self.signal_mid_point
        perc = self.overlap_net(idx.unsqueeze(-1) / self.signal_len * 2 - 1).flatten()
        return (x_i, perc)


def make_find_window_configs(idx_to_window: IdxToWindow, last_sample: int):
    """
    Creates a function which scans the mapping function to generate window
    positions and overlaps ranging from just before the first sample to
    just after the last sample.
    """
    prev_cached_i = 0

    def find_window_configs():
        nonlocal prev_cached_i
        # evaluate the window generator until we hit boundary on both sides,
        # but keep one extra element on both sides past boundary

        eval_cache = {}

        def fast_idx_to_window(i):
            batch_size = 16
            idx = i // batch_size
            arr = eval_cache.get(idx)
            if arr is not None:
                return (arr[0][i - idx * batch_size], arr[1][i - idx * batch_size])
            arr = idx_to_window(torch.arange(start=idx * batch_size, end=idx * batch_size + batch_size,
                                             step=1, dtype=torch.float32, device="cuda", requires_grad=False))
            eval_cache[idx] = arr
            return (arr[0][i - idx * batch_size], arr[1][i - idx * batch_size])

        window_configs = []
        i = prev_cached_i
        w, p = fast_idx_to_window(i)
        if w >= last_sample:
            path = 0
            # move left if too big
            while w >= last_sample:
                i = i - 1
                w, p = fast_idx_to_window(i)
            prev_cached_i = i
            window_configs.append(fast_idx_to_window(i + 1))
            # collect all items between last_sample and 0
            while w >= 0:
                window_configs.append((w, p))
                i = i - 1
                w, p = fast_idx_to_window(i)
            window_configs.append((w, p))
            window_configs.reverse()
        elif w <= 0:
            path = 1
            # move right if too small
            while w <= 0:
                i = i + 1
                w, p = fast_idx_to_window(i)
            window_configs.append(fast_idx_to_window(i - 1))
            prev_cached_i = i
            # collect all items between 0 and last_sample
            while w <= last_sample:
                window_configs.append((w, p))
                i = i + 1
                w, p = fast_idx_to_window(i)
            window_configs.append((w, p))
        else:
            path = 2
            # w was in range
            right_list = []
            while w <= last_sample:
                right_list.append((w, p))
                i = i + 1
                w, p = fast_idx_to_window(i)
            right_list.append((w, p))
            # move left from zero, to cover the starting regions
            i = prev_cached_i - 1
            w, p = fast_idx_to_window(i)
            left_list = []
            while w >= 0:
                left_list.append((w, p))
                i = i - 1
                w, p = fast_idx_to_window(i)
            left_list.append((w, p))
            left_list.reverse()
            window_configs = left_list + right_list
        # filter out windows that are too small
        filt_window_configs = []
        prev_window_sample = -math.inf
        assert window_configs[0][0] < 0, f"{i} {path} {window_configs}"
        assert window_configs[-1][0] > last_sample, f"{i} {path} {window_configs}"
        # min window size is 1
        for i in range(len(window_configs)):
            if i > 0:
                if not (window_configs[i][0] > window_configs[i - 1][0]):
                    if path == 2:
                        path_desc = f"({(len(left_list), len(right_list))})" # type: ignore
                    else:
                        path_desc = None
                    assert False, f"path: {path} {path_desc}, i: {i}, {window_configs}, {window_configs[i][0]} > {window_configs[i - 1][0]}"
            if window_configs[i][0] - prev_window_sample > 1:
                filt_window_configs.append(window_configs[i])
                prev_window_sample = window_configs[i][0]
        assert len(filt_window_configs) > 0, f"{len(window_configs)}, path: {path}, prev_cached_i: {prev_cached_i}, {fast_idx_to_window(prev_cached_i)}, {fast_idx_to_window(prev_cached_i - 1)}, {idx_to_window.slope}"
        assert filt_window_configs[0][0] < 0, f"{i} {path} {filt_window_configs}"
        assert filt_window_configs[-1][0] > last_sample, f"{i} {path} {filt_window_configs}"
        assert len(filt_window_configs) > 1, f"{filt_window_configs}, slope: {idx_to_window.slope.item()}"
        return filt_window_configs

    return find_window_configs
