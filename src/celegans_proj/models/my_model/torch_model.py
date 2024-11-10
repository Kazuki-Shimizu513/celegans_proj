
import math
import sys
import os

import numpy as np  

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    basic_modules,
)







class MyTorchModel(nn.Module):

    def __init__(
            self, 
            num_in_ch=1, 
            size=4, 
            num_layers = 4, 
            inplace=True

    ) -> None:
        super().__init__()

        self.size = size

        keep_stats = True

        out_channels = 16

    def forward(self, img):

        x = self.model(img) # B, 128, W/16, H/16

        return x


