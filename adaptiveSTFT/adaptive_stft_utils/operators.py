import torch.autograd
import torch
import torch.nn.functional as F


def dithering_int(n):
    if n == int(n):
        return int(n)
    return int(torch.bernoulli((n - int(n))) + int(n))


class SignPassGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, n):
        return torch.sign(n)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 1e-3


class InvSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, n):
        return torch.sign(n)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output * 1e-3


def clip_tensor_norm(parameters, max_norm, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].device
    total_norm = torch.norm(torch.stack([torch.norm(p.detach(), norm_type).to(
        device) for p in parameters]), norm_type).detach()

    def clamp(p):
        clamped = torch.clamp(p, min=-total_norm * max_norm, max=total_norm * max_norm)
        return clamped + 1e-4 * (p - clamped)
    return [
        clamp(p)
        for p in parameters
    ]


class LimitGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = clip_tensor_norm(grad_output, max_norm=1.0, norm_type=2)[0]
        return grad_output

