
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .wddd2_ad_config import Config 
from celegans_proj.third_party.SimSID.models import squid
from celegans_proj.third_party.SimSID.models.memory import MemoryQueue
from celegans_proj.third_party.SimSID.tools import (
        parse_args, 
        build_disc, 
        log, 
        log_loss, 
        save_image, 
        backup_files
    )



class SimSIDModel(nn.Module):
    """SimSID Module

    Args:

    """
    def __init__(
        self,
        CONFIG = None,
        **kwargs,
    ) -> None:
        super().__init__()

        CONFIG = Config() 
        self.model = squid.AE(
            CONFIG, 
            features_root=32, 
            level=CONFIG.level
        )
        # self.discriminator  = build_disc(CONFIG)

    def  forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor] :
        """ forward pass images to network

        Args:
            images (torch.Tensor) : Batch of images

        Returns:
            output (torch.Tensor | list[torch.Tensor] | tuple[list[torch.Tensor]]) : 
        """

        outs = model(images)

        # outs = dict(recon=whole_recon, embeddings=embeddings)
        # if self.dist:
        #     outs['teacher_recon'] = t_x
        #     outs['dist_loss'] = self_dist_loss
        # if self.config.analyze_memory:
        #     outs['alpha'] = alpha
 
        return outs

if __name__ == "__main__":
    pass
