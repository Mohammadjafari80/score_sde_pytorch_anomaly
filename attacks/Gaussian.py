
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)

        adv_images = images + torch.randn(images.size(), device=device) * self.std + self.mean
        return adv_images

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images