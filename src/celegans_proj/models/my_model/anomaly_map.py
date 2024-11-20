
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F  

from torchmetrics.functional.image import (
    peak_signal_noise_ratio as PSNR,
)


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.

    Args:
        image_size (ListConfig, tuple): Size of original image used for upscaling the anomaly map.
        sigma (int): Standard deviation of the gaussian kernel used to smooth anomaly map.
            Defaults to ``4``.

    Raises:
        ValueError: In case modes other than multiply and add are passed.
    """

    def __init__(
        self,
        image_size: tuple = (256,256),
    ) -> None:
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)

    def forward(
        self, 
        img, pred_img, pred_mask
    ) -> torch.Tensor:
        """Compute anomaly map given encoder and decoder features.

        Args:
            student_features (list[torch.Tensor]): List of encoder features
            teacher_features (list[torch.Tensor]): List of decoder features

        Returns:
            Tensor: Anomaly maps of length batch.
        """


        anomaly_map = self.compute_recon_distance(
                        pred_img, img, 
                        mode = "L2", 
                        perceptual = False, 
                      )

        # TODO :: Class Histgram Maharanobis Distance using pred_mask


        pred_score = self.compute_metrics(pred_img, img, pred_mask, anomaly_map, mode="psnr")

        output =  {"anomaly_map": anomaly_map, "pred_score": pred_score}
        return output 

    @staticmethod
    def compute_recon_distance(
        pred, target,
        mode = "L2", 
        perceptual = False, 
    ):

        assert pred.shape == target.shape,\
            f"{pred.shape=}\t{target.shape=}"

        diffs=0
        if mode=="L1":
           # https://note.nkmk.me/python-opencv-numpy-image-difference/
           diffs = torch.abs(pred - target) 
        elif mode == "L2":
           diffs = torch.sqrt((pred - target)**2) 
        else:
            print("please check mode attr in calc difference method")

        assert diffs.shape == target.shape,\
            f"{diffs.shape=}\t{target.shape=}"


        return diffs

    def compute_metrics(
        self,
        pred_imgs, imgs,
        pred_masks,
        anomaly_maps,
        mode = "psnr", 
    ):
        # Calc_psnr metrics
        scores = []
        for idx, (prd, img, amp, msk) in enumerate(zip(
                pred_imgs, imgs, anomaly_maps, pred_masks
        )):
            if mode == "psnr":
                score = PSNR(prd, img)
            else:
                score = torch.max(amp) # anomaly_map
            scores.append(torch.Tensor(score))
        scores = torch.stack(scores, dim=0)
        return scores






if __name__ == "__main__":
    print()
