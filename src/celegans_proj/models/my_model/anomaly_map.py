
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F  

from torchmetrics.functional.image import (
    peak_signal_noise_ratio as PSNR,
)
from torchmetrics.functional.segmentation import mean_iou

class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.
    """

    def forward(
        self, 
        outputs,
    ) -> torch.Tensor:
        """Compute anomaly map given encoder and decoder features.

        Args:
            outputs : dict of result of batch iter

        Returns:
            Tensor: Anomaly maps of length batch.
        """


        anomaly_maps = self.compute_recon_distance(
                        outputs["pred_imgs"], outputs["image"], 
                        mode = "L2", 
                      )

        # TODO :: Cosine Distance using wildtype feature and test feature cropped by pred_mask
        # TODO :: or Class Histgram Maharanobis Distance using pred_mask

        pred_scores = self.compute_metrics(
                        outputs["pred_imgs"], outputs["image"], 
                        outputs['pred_masks'], outputs['gen_masks'], 
                        anomaly_maps, 
                        mode=None,
        )

        return anomaly_maps, pred_scores 

    @staticmethod
    def compute_recon_distance(
        pred, target,
        mode = "L2", 
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
        pred_masks, gen_masks, 
        anomaly_maps=None,
        mode = None,# "psnr", 
    ):
        """
        Return:
            pred_score : (normal)0-1(anomaly)
        """

        assert pred_imgs.shape == imgs.shape,\
            f"{pred_imgs.shape=}\t{imgs.shape=}"

        # Calc reconstruction metrics
        scores = []
        if mode == "psnr":
            for pred, img in zip(pred_imgs, imgs):
                score = PSNR(pred, img, )
                scores.append(score)
        else:
            for ano_map in anomaly_maps:
                score = torch.max(ano_map)
                scores.append(score)
        scores = torch.stack(scores, dim=0)
        # print(f"recon  value : {scores},\n\t{scores.shape=}")
        # Calc segmentation metrics
        _scores = []
        for pred, gen in zip(pred_masks, gen_masks):
            mIoU = self.calc_meanIoU(pred, gen,)
            score = 1 - mIoU.mean()
            _scores.append(score)
        _scores = torch.stack(_scores, dim=0)
        # print(f"segment value : {_scores},\n\t{_scores.shape=}")
 
        scores = (scores + _scores)/2

        return scores

    def min_max_scaling(self,v, new_min=0.0, new_max=1.0):
        v_min, v_max = v.min(), v.max()
        v_p = (v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        return v_p

    @staticmethod
    def calc_meanIoU(pred, gen, ):
        if pred.shape[1] >= gen.shape[1]:
            num_classes = pred.flatten().unique().size(dim=0)# .item()
        else:
            num_classes = gen.flatten().unique().size(dim=0)# .item()

        pred_one_hot = torch.nn.functional.one_hot(pred.unsqueeze(0), num_classes=num_classes)
        # Convert from NHWC to NCHW
        pred_one_hot = pred_one_hot.permute(0, 3, 1, 2)# .type(torch.float)

        gen_one_hot = torch.nn.functional.one_hot(gen.unsqueeze(0), num_classes=num_classes)
        # Convert from NHWC to NCHW
        gen_one_hot = gen_one_hot.permute(0, 3, 1, 2)# .type(torch.float)

        mIoU = mean_iou(
            pred_one_hot, gen_one_hot,
            num_classes=num_classes,
            include_background=True, 
            per_class=False, 
            input_format='one-hot'
        )

        return mIoU






if __name__ == "__main__":
    print()
