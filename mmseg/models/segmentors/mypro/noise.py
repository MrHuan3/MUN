import torch
from torch import nn

class Noise(nn.Module):
    def __init__(self):
        super(Noise, self).__init__()
        self.in_rgb_mean = [0.485, 0.456, 0.406]
        self.in_rgb_std = [0.229, 0.224, 0.225]
        self.my_rgb_mean = [0.459, 0.448, 0.409]
        self.my_rgb_std = [0.230, 0.226, 0.240]
        self.diff_rgb_mean = [self.in_rgb_mean[i] - 
                              self.my_rgb_mean[i] 
                              for i in range(3)]
        self.diff_rgb_std = [self.in_rgb_std[i] + 
                             self.my_rgb_std[i]
                             for i in range(3)]
        
        
    def forward(self, rgb, gt):
        pos = torch.tile(gt.to(bool), (1, 3, 1, 1))
        neg = torch.tile(torch.where(gt.to(bool), False, True), (1, 3, 1, 1))
        noise_pos = torch.tile(torch.rand(gt.shape), (1, 3, 1, 1)).to(rgb.device)
        noise_neg = torch.tile(torch.rand(gt.shape), (1, 3, 1, 1)).to(rgb.device)
        noise_pos = torch.where(pos, noise_pos, 0)
        noise_neg = torch.where(neg, noise_neg, 0)
        for i in range(3):
            noise_pos[:,i,:,:] *= self.diff_rgb_mean[i]
            noise_neg[:,i,:,:] *= self.diff_rgb_mean[i]
        noised_rgb = rgb + noise_pos + noise_neg
        
        return noised_rgb
