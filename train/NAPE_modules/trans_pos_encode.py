import os
import sys
import random
import math
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pathlib import Path

from .NAPE import Position_encode
root = f'{Path.home()}/codes/'

class TT_Pos_Encode(nn.Module):
    """
        Uses the learned position encoding as the ordering for this
        Sinusoidal vector positional encoding.
    """

    def __init__(self, hidden_size, N, d, PE_name='', device=None, scale=110000.0):
        super(TT_Pos_Encode, self).__init__()

        self.PE_name = PE_name
        self.N = N
        self.d = d
        self.hidden_size = hidden_size
        self.device = device
        self.scale = scale


    def get_ordering(self, N, d):
        """This would return the saved position encoding"""
        PE = Position_encode(N=N, d=d)

        PE.load_state_dict(torch.load(root+'NAPE-wt-node2vec/Saved_models/'+f'd={d}_{self.PE_name}'))
        PE.eval()
        Z,_,_,_ = PE(test=True)
        err = 0.0001
        Z = Z + err
        Z = torch.round(Z)

        # We convert binary now to decimal
        return self.convert_bin_to_dec(Z)


    def convert_bin_to_dec(self, Z):
        """Function returns the decimal equivalent of the binary input"""
        d = self.d
        two_vec = torch.zeros(d)
        for i in range(d):
            two_vec[i] = pow(2,d-1-i)
        numbers = (Z * two_vec).sum(dim=1)

        return numbers.detach()


    def inv_timescales_fn(self, hidden_size):
        """Time scale for positional encoding"""

        num_timescales = hidden_size // 2
        max_timescale = self.scale # change from 10000 to 110000
        min_timescale = 1.0

        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1))

        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)

        return inv_timescales


    def get_position_encoding(self):
        """The position encoding is computed and returned"""

        ordering = self.get_ordering(self.N, self.d)
        inv_timescales = self.inv_timescales_fn(self.hidden_size)

        max_length = self.N #number of nodes
        position = ordering
        if self.device is not None:
            position.to(self.device)

        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
        signal = signal.squeeze(0) #shape:([N, hidden_size])

        return signal #shape:([N, hidden_size])
