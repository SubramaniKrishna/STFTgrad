import math
from tqdm import trange
import sys
import pathlib
import torch.autograd
import torch
import numpy as np
import torch.optim
import torch.nn as nn
from celluloid import Camera
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch.nn.functional as F
from adaptive_stft_utils.operators import dithering_int, Sign, InvSign
from adaptive_stft_utils.mappings import IdxToWindow, make_find_window_configs
from adaptive_stft_utils.losses import kurtosis


class NumericalError(Exception):
    def __init__(self, message, grad_hist=None, window_times_signal_grads=None, f_grad=None):
        self.message = message
        self.grad_hist = grad_hist
        self.window_times_signal_grads = window_times_signal_grads
        self.f_grad = f_grad

    def __str__(self):
        if self.message:
            return 'NumericalError, {0} '.format(self.message)
        else:
            return 'NumericalError'


#COLOR_MAP = 'GnBu'
COLOR_MAP = None


def optimize_stft(
    s,
    lr=1e-4,
    num_windows=None,
    sgd=False,
    num_epochs=9000,
    score_fn=kurtosis,
    window_shape='trapezoid',
    make_animation=True,
    name_for_saving='',
):
    if window_shape not in ['trapezoid', 'triangle']:
        raise RuntimeError(f'Unknown window shape {window_shape}')

    kur_hist = []
    fig_width = int(s.size(0) / 27000 * 14 * 10) / 10
    anim_fig = Figure(figsize=(fig_width, 7))
    preview_fig = Figure(figsize=(fig_width, 7))
    from IPython.core.display import display
    preview_handle = display(preview_fig, display_id=True)
    camera = Camera(anim_fig)
    matrix_fig = plt.figure(figsize=(22, 15))
    s = s.cuda()

    # make (num_windows - 1) points, in addition to start and end (0, signal_len)
    assert len(s.shape) == 1
    last_sample = s.size(0)
    if num_windows is None:
        assert s.size(0) > 512
        num_windows = s.size(0) // 512
    idx_to_window = IdxToWindow(
        signal_len=last_sample, num_windows=num_windows)
    idx_to_window = idx_to_window.cuda()

    if sgd:
        optimizer = torch.optim.SGD(idx_to_window.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(
            idx_to_window.parameters(), lr=lr, amsgrad=True, weight_decay=1e-6)

    find_window_configs = make_find_window_configs(idx_to_window, last_sample=last_sample)

    xs = None
    ys = None
    with trange(num_epochs + 1) as t:
        for ep in t:
            optimizer.zero_grad()

            # window_configs excludes 0 and sample_length
            configs = find_window_configs()
            xs, ys, s_ext, extend_left = make_window_extend_signal(configs, s, window_shape=window_shape)

            rffts_not_detached = []
            rffts = []
            len_xs = xs.size(0)
            assert len_xs > 1, f"xs: {xs}, ys: {ys}, slope: {idx_to_window.slope.item()}"
            for i in range(len_xs - 1):
                rfft = apply_adaptive_window(s_ext, xs[i], ys[i], xs[i + 1], ys[i + 1], window_shape=window_shape)
                rfft_sq = rfft[..., 0] ** 2 + rfft[..., 1] ** 2
                rffts_not_detached.append(rfft_sq)
                rffts.append(rfft_sq.detach().cpu().numpy())

            n_wnd = len(rffts_not_detached)
            score = score_fn(rffts_not_detached)
            t.set_postfix(score=score.item(),
                          slope=idx_to_window.slope.item(), n_wnd=n_wnd)

            if (torch.isnan(score).any() or torch.isinf(score).any()):
                raise NumericalError(
                    f'score become NaN at iteration {ep}')

            (-score).backward()
            torch.nn.utils.clip_grad_norm_(
                idx_to_window.parameters(), max_norm=1)
            optimizer.step()

            kur_hist.append(score.item())

            plot_width = int(768 * fig_width / 7)

            def get_scaled_fft_plots():
                # since each window has different size, stretch the FFT frequency to fit the largest
                max_size = np.max([x.shape[0] for x in rffts])
                scaled_fft_plots = np.zeros(
                    (max_size, last_sample), dtype=np.float32)
                i = 0
                from scipy.interpolate import interp1d
                for i, fft in enumerate(rffts):
                    bins = np.linspace(0, max_size, fft.shape[0])
                    f_out = interp1d(bins, fft, axis=0, kind='nearest')
                    new_bins = np.linspace(0, max_size, max_size)
                    fft_out = f_out(new_bins)
                    fft_out /= np.max(fft_out)
                    if i == 0:
                        start_point = int(max(ys[i] - extend_left, 0))
                    else:
                        start_point = int(max((xs[i] + ys[i]) / 2 - extend_left, 0))
                    if i < len(rffts) - 1:
                        end_point = int((xs[i + 1] + ys[i + 1]) / 2) - extend_left
                    else:
                        end_point = last_sample
                    scaled_fft_plots[:, start_point:end_point] = np.expand_dims(fft_out, -1)
                import cv2
                scaled_fft_plots = np.power(scaled_fft_plots, 0.5)
                return cv2.resize(scaled_fft_plots, dsize=(plot_width, 768))

            if ep % (num_epochs // 8) == 0:
                outfile = f'{name_for_saving}_plot_data_{ep}.npz'
                model_path = f'{name_for_saving}_mapping_model_{ep}.pth'
                scaled_fft_plots = get_scaled_fft_plots()
                np.savez(outfile, spectro=scaled_fft_plots, 
                                  x=xs.cpu().detach().numpy(),
                                  y=ys.cpu().detach().numpy(),
                                  extend_left=extend_left,
                                  sample_length=last_sample,
                                  sample=s.cpu().detach().numpy(),
                                  sample_extended=s_ext.cpu().detach().numpy())
                torch.save(idx_to_window.state_dict(), model_path)
                plt.gcf().add_subplot(3, 3, ep // (num_epochs // 8) + 1)
                import matplotlib.colors
                plt.gca().pcolormesh(scaled_fft_plots, norm=matplotlib.colors.Normalize(), linewidth=0, cmap=COLOR_MAP)
                plt.gca().set_title(f'ep: {ep}, score: {score.item():.5f}')
                for i in range(xs.size(0)):
                    inter_window_line = (xs[i] + ys[i]).item() / 2 - extend_left
                    if inter_window_line <= 0 or inter_window_line >= last_sample:
                        continue
                    plt.gca().axvline(inter_window_line / last_sample * plot_width,
                                      linewidth=0.5, antialiased=True)

            if ep % 15 == 0:
                scaled_fft_plots = get_scaled_fft_plots()

                def draw(fig):
                    fig.gca().pcolormesh(scaled_fft_plots, norm=matplotlib.colors.Normalize(), linewidth=0, cmap=COLOR_MAP)
                    fig.gca().text(0.3, 1.01, f'ep: {ep}, score: {score.item():.5f}', transform=fig.gca().transAxes)
                    for i in range(xs.size(0)):
                        inter_window_line = (xs[i] + ys[i]).item() / 2 - extend_left
                        if inter_window_line <= 0 or inter_window_line >= last_sample:
                            continue
                        fig.gca().axvline(inter_window_line / last_sample * plot_width,
                                          linewidth=0.5, antialiased=True)
                if make_animation:
                    draw(anim_fig)
                    camera.snap()
                preview_fig.gca().clear()
                draw(preview_fig)
                # Show image on notebook
                preview_fig.canvas.draw()
                preview_handle.update(preview_fig)
            
            if ep % 30 == 0:
                import gc
                gc.collect()

    if make_animation:
        ani = camera.animate(interval=33.3, blit=True)
    else:
        ani = None
    preview_handle.update(plt.figure())
    matrix_fig.tight_layout()
    return idx_to_window, kur_hist, ani


def make_window_extend_signal(configs, s: torch.Tensor, window_shape: str):
    last_sample: int = s.size(0)
    xs = [x for (x, _) in configs]
    xs[0] = torch.clamp(xs[0], -last_sample + 1, 0)
    xs[-1] = torch.clamp(xs[-1], 0, last_sample * 2 - 1)
    if window_shape == 'trapezoid':
        ys = [xs[i + 1] - (xs[i + 1] - xs[i]) * configs[i][1]
            for i in range(len(xs) - 1)]
        ys.insert(0, xs[0] - (xs[1] - xs[0]) * configs[0][1])
    elif window_shape == 'triangle':
        ys = [xs[i] for i in range(len(xs))]
        # pick x values that are in sample range
        xs.pop(0)
    else:
        raise RuntimeError(f'Unknown window shape {window_shape}')
    ys[0] = torch.clamp(ys[0], -last_sample + 1, 0)
    xs = torch.cat([x.view(1) for x in xs])

    # extend the signal both ways via zero padding
    offset = -int(torch.floor(ys[0]))
    assert offset >= 0
    extend_left = offset
    assert extend_left <= last_sample
    extend_right = int(torch.ceil(xs[-1])) - last_sample + 1
    assert extend_right >= 0
    assert extend_right <= last_sample
    s_left_pad = torch.zeros_like(s[:extend_left])
    s_right_pad = torch.zeros_like(s[last_sample - extend_right:])
    s_ext = torch.cat((s_left_pad, s, s_right_pad))

    xs = xs + extend_left
    ys = torch.cat([y.view(1) for y in ys])
    ys = ys + extend_left
    return xs, ys, s_ext, extend_left


def apply_adaptive_window(
    s_ext: torch.Tensor,
    x_i: torch.Tensor,
    y_i: torch.Tensor,
    x_next: torch.Tensor,
    y_next: torch.Tensor,
    window_shape: str
) -> torch.Tensor:
    if window_shape == 'trapezoid':
        # three parts
        left_trig_start = dithering_int(y_i)
        left_trig_end = dithering_int(x_i)
        right_trig_start = dithering_int(y_next)
        right_trig_end = dithering_int(x_next)
        rect_start = left_trig_end
        rect_end = right_trig_start
        m = torch.arange(0, left_trig_end - left_trig_start,
                         dtype=torch.float32, device=s_ext.device)
        ramp = m / (x_i - y_i)
        left_ramp_times_signal = ramp * \
            s_ext[left_trig_start:left_trig_end]
        m = torch.arange(0, right_trig_end - right_trig_start,
                         dtype=torch.float32, device=s_ext.device)
        ramp = 1 - (m / (x_next - y_next))
        right_ramp_times_signal = ramp * s_ext[right_trig_start:right_trig_end]
        rect_signal = s_ext[rect_start:rect_end]
        # rect_signal = rect_signal * Sign.apply(y_next) * InvSign.apply(x_i)
        window_times_signal = torch.cat(
            (left_ramp_times_signal, rect_signal, right_ramp_times_signal), dim=-1)

    elif window_shape == 'triangle':
        left_trig_start = dithering_int(y_i)
        left_trig_end = dithering_int(x_i)
        right_trig_start = dithering_int(x_i)
        right_trig_end = dithering_int(x_next)
        m = torch.arange(0, left_trig_end - left_trig_start,
                         dtype=torch.float32, device=s_ext.device)
        ramp = m / (x_i - y_i)
        left_ramp_times_signal = ramp * \
            s_ext[left_trig_start:left_trig_end]
        m = torch.arange(0, right_trig_end - right_trig_start,
                         dtype=torch.float32, device=s_ext.device)
        ramp = 1 - (m / (x_next - y_next))
        right_ramp_times_signal = ramp * s_ext[right_trig_start:right_trig_end]
        window_times_signal = torch.cat(
            (left_ramp_times_signal, right_ramp_times_signal), dim=-1)

    else:
        raise RuntimeError(f'unknown window shape {window_shape}')

    assert window_times_signal.size(0) > 1, f"x: {x_i}, y: {y_i}"
    rfft = torch.rfft(window_times_signal,
                    signal_ndim=1, normalized=True)
    return rfft
