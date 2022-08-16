tensor + torch.randn(tensor.size()) * self.std + self.mean

import torch
import torch.nn as nn
from attacks.attack import Attack

class Gaussian():
    def __init__(self, model, mean=0.0, std=0.02):
        super().__init__("FGSM", model)
        self.mean = mean
        self.std = std

    def forward(self, images, labels):
        """
        Overridden.
        """
        images = images.clone().detach().to(self.device)

        adv_images = images + torch.randn(images.size()) * self.std + self.mean

        return adv_images
