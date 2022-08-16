
import torch
import torch.nn as nn

class Gaussian():
    def __init__(self, model, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def forward(self, images, labels):
        """
        Overridden.
        """
        adv_images = images + torch.randn(images.size()) * self.std + self.mean
        return adv_images

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images