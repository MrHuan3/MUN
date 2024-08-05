import torch
from torch import nn


class ImgNorm(nn.Module):

    def __init__(self):
        super().__init__()
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)

    def forward(self, inputs):
        list = []
        for idx in range(inputs.shape[0]):
            list.append(inputs[idx,:,:,:])

        list = [(_input - self.mean) / self.std for _input in list]
        
        for idx in range(len(list)):
            inputs[idx,:,:,:] = list[idx]
        return inputs
