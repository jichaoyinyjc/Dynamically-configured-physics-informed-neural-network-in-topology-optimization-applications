import rff
import torch
import torch.nn as nn
import numpy as np
import random

random.seed(10000)
np.random.seed(10000)
torch.manual_seed(10000)
torch.cuda.manual_seed(10000)


class NN1(torch.nn.Module):
    def __init__(self, neurons_per_layer, n_layers, rff_std, act_fn):
        super(NN1, self).__init__()
        d_in, d_hidden, d_out = (int(neurons_per_layer[0]), int(neurons_per_layer[1]), int(neurons_per_layer[2]))
        self.n_layers = int(n_layers)
        self.act_fn = act_fn

        # input
        self.input = rff.layers.GaussianEncoding(sigma=rff_std, input_size=d_in, encoded_size=d_hidden // 2)
        # hidden
        self.net = torch.nn.ModuleList()
        for ii in range(self.n_layers):
            self.net.append(torch.nn.Linear(d_hidden, d_hidden))
            torch.nn.init.constant_(self.net[-1].bias, 0.)
            torch.nn.init.normal_(self.net[-1].weight, mean=0., std=np.sqrt(1. / d_hidden))
        # output
        self.output = torch.nn.Linear(d_hidden, d_out)
        torch.nn.init.constant_(self.output.bias, 0.)
        torch.nn.init.normal_(self.output.weight, mean=0., std=np.sqrt(1. / d_hidden))

    def forward(self, x):
        activation_fn = getattr(torch, self.act_fn)
        # input, transform x to the frequency domain as input
        x = self.input(x)
        # hidden Layer - 1
        for id_ in range(self.n_layers):
            x = activation_fn(self.net[id_](x))
        # output
        y = self.output(x)
        return y


class Block(nn.Module):

    def __init__(self, neural):
        super(Block, self).__init__()
        self.act = torch.tanh
        self.layer_1 = nn.Linear(neural, neural)
        torch.nn.init.constant_(self.layer_1.bias, 0.)
        torch.nn.init.normal_(self.layer_1.weight, mean=0., std=np.sqrt(1. / neural))
        self.layer_2 = nn.Linear(neural, neural)
        torch.nn.init.constant_(self.layer_2.bias, 0.)
        torch.nn.init.normal_(self.layer_2.weight, mean=0., std=np.sqrt(1. / neural))
        self.layer_norm_1 = nn.LayerNorm(neural)
        self.layer_norm_2 = nn.LayerNorm(neural)

    def forward(self, x):
        identity = x
        out = self.layer_norm_1(x)
        out = self.act(self.layer_1(out))
        out = self.layer_norm_2(out)
        out = self.layer_2(out) + identity
        out = self.act(out)
        return out


class NN2(torch.nn.Module):
    def __init__(self, neurons_per_layer, n_layers, act_fn):
        super(NN2, self).__init__()
        d_in, d_hidden, d_out = (int(neurons_per_layer[0]), int(neurons_per_layer[1]), int(neurons_per_layer[2]))
        self.n_layers = int(n_layers)
        self.act_fn = act_fn

        # input
        self.input = torch.nn.Linear(d_in, d_hidden)
        torch.nn.init.constant_(self.input.bias, 0.)
        torch.nn.init.normal_(self.input.weight, mean=0., std=np.sqrt(1. / d_hidden))
        # hidden
        self.net = torch.nn.ModuleList()
        for ii in range(self.n_layers):
            self.net.append(Block(d_hidden))
        # output
        self.output = torch.nn.Linear(d_hidden, d_out)
        torch.nn.init.constant_(self.output.bias, 0.)
        torch.nn.init.normal_(self.output.weight, mean=0., std=np.sqrt(1. / d_hidden))

    def forward(self, x):
        x = self.input(x)
        # hidden Layer - 1
        for id_ in range(self.n_layers):
            x = self.net[id_](x)
        # output
        y = self.output(x)
        return y
