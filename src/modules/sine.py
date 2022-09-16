import torch
from torch import nn


class Sine(nn.Module):
    def __init__(self, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, input):
        return torch.sin(self.omega_0 * input)
