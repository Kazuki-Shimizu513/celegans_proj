
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F  

from torchmetrics.functional.image import (
    peak_signal_noise_ratio as PSNR,
)


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.
    """

    def forward(
        self, 
        imgs, pred_imgs, pred_masks=None,
    ) -> torch.Tensor:
        """Compute anomaly map given encoder and decoder features.

        Args:
            student_features (list[torch.Tensor]): List of encoder features
            teacher_features (list[torch.Tensor]): List of decoder features

        Returns:
            Tensor: Anomaly maps of length batch.
        """


        anomaly_maps = self.compute_recon_distance(
                        pred_imgs, imgs, 
                        mode = "L2", 
                        perceptual = False, 
                      )

        # TODO :: Cosine Distance using wildtype feature and test feature cropped by pred_mask
        # TODO :: or Class Histgram Maharanobis Distance using pred_mask

        pred_scores = self.compute_metrics(
                          pred_imgs, imgs, 
                          anomaly_maps, 
                          pred_masks, 
                          mode="psnr",
        )

        output =  {"anomaly_map": anomaly_maps, "pred_score": pred_scores}
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
        anomaly_maps=None,
        pred_masks=None,
        mode = "psnr", 
    ):

        assert pred_imgs.shape == imgs.shape,\
            f"{pred_imgs.shape=}\t{imgs.shape=}"



        if mode == "psnr":
            # Calc_psnr metrics
            scores = []
            for pred, img in zip(pred_imgs, imgs):
                score = PSNR(pred, img, )
                scores.append(score)
            scores = torch.stack(scores, dim=0)
        else:
            scores = torch.max(anomaly_maps, dim=0)

        # TODO:: add metric for segmentation mask 

        return scores





if __name__ == "__main__":
    print()
