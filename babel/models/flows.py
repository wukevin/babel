import numpy as np

import torch
import torch.nn as nn

import skorch

def make_checkerboard_vec(n, start_zero=False):
    """Make a checkerboard pattern of n items as a float array"""
    retval = np.array([(i % 2) == 0 for i in range(n)]).astype(np.float32)
    if start_zero:
        retval = 1.0 - retval
    return retval

def make_checkerboard_mask(num_layers:int, dim:int):
    """Make the list of masks"""
    x = make_checkerboard_vec(dim)
    return np.array([list(x), list(1.0 - x)] * num_layers)

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()

def gen_fc_net(num_inputs, num_hidden, act=nn.LeakyReLU, final_act=None) -> nn.Sequential:
    """
    Generates a fully connected network of size [input, hidden, hidden, input]
    """
    layers = [
        nn.Linear(num_inputs, num_hidden), act(),
        nn.Linear(num_hidden, num_hidden), act(),
        nn.Linear(num_hidden, num_inputs),
    ]
    if final_act is not None:
        layers.append(final_act())

    return nn.Sequential(*layers)

class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """
    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data * (1 - self.momentum))
                self.running_var.add_(self.batch_var.data * (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)

class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py
    """
    def __init__(self, num_inputs, num_hidden, mask=None, act=nn.LeakyReLU, s_act_func=nn.Tanh, t_act_func=None):
        super(CouplingLayer, self).__init__()
        self.num_inputs = num_inputs
        if mask is None:
            mask = (torch.arange(0, num_inputs) % 2).type(torch.float32)
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.scale_net = gen_fc_net(num_inputs, num_hidden, act=act, final_act=s_act_func)
        self.translate_net = gen_fc_net(num_inputs, num_hidden, act=act, final_act=t_act_func)

    def forward(self, x, cond_inputs=None, mode='direct'):
        mask = self.mask
        inv_mask = (1. - mask)
        masked_x = x * mask

        log_s = self.scale_net(masked_x) * inv_mask
        t = self.translate_net(masked_x) * inv_mask
        if mode == 'direct':
            z = inv_mask * (x - t) * torch.exp(-log_s) + masked_x
        else:
            z = inv_mask * (x * torch.exp(log_s) + t) + masked_x
        return z, log_s.sum(dim=1, keepdim=False)

    def sample(self, num_samples, noise=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        samples = self.forward(noise, mode='inverse')[0]
        return samples

class RealNVP(nn.Module):
    """
    RealNVP implementation

    Notes:
    - For a transformation to be invertible, it must be from R^d -> R^d
    - Composition of invertible transformations is invertible as well
    """
    def __init__(self, num_blocks:int, num_inputs:int, num_hidden:int, act_func=nn.LeakyReLU, s_act_func=nn.Tanh, t_act_func=None, use_batchnorm:bool=False):
        torch.manual_seed(59239)
        super(RealNVP, self).__init__()
        assert (num_blocks % 2) == 0, f"Number of blocks should be even but got {num_blocks}"
        self.num_inputs = num_inputs

        self.blocks = nn.ModuleList()
        mask = (torch.arange(0, num_inputs) % 2).type(torch.float32)
        for _i in range(num_blocks):
            self.blocks.append(CouplingLayer(num_inputs, num_hidden, mask=mask, act=act_func, s_act_func=s_act_func, t_act_func=t_act_func))
            if use_batchnorm:
                self.blocks.append(BatchNormFlow(num_inputs))
            mask = 1.0 - mask  # Alternate

    def forward(self, x, mode='direct', logdets=None):
        if logdets is None:
            logdets = torch.zeros(x.size(0), device=x.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self.blocks:
                x, logdet = module(x, mode=mode)
                logdets -= logdet
        else:
            for module in reversed(self.blocks):
                x, logdet = module(x, mode=mode)
                logdets -= logdet

        return x, logdets

    def sample(self, num_samples, noise=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.blocks[0].parameters()).device
        noise = noise.to(device)
        samples = self.forward(noise, mode='inverse')[0]
        return samples

class FlowSkorchNet(skorch.NeuralNet):
    """Subclassed so that we can easily extract the encoded layer"""
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        return super().get_loss(y_pred, y_true, *args, **kwargs)

    def sample(self, num_samples, noise=None):
        return self.module_.sample(num_samples=num_samples, noise=noise)

