

import torch as th
from torch import nn

from .components import DiffSeg

class MyDiffSegParamNet(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        self.segmentor = DiffSeg(

        )

    def forward(self,attn_maps):

        return pred_masks



if __name__ == "__main__":
    pass
