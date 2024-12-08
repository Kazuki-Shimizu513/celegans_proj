
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F  

from torchmetrics.functional.image import (
    peak_signal_noise_ratio as PSNR,
    multiscale_structural_similarity_index_measure as ms_ssim,
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

        # image = self.cvt_channel_gray2RGB(outputs["image"])
        recon_dist = self.compute_recon_distance(
                        outputs["pred_imgs"],  outputs["image"],# image,# 
                        mode = "L2", 
                      )

        # TODO :: Cosine Distance using wildtype feature and test feature cropped by pred_mask
        seg_dist = self.compute_segmentation_distance(
                        outputs["pred_masks"], 
                        outputs["pred_latents"], outputs["gen_latents"], 
                        mode = "cosine", 
                      )

        anomaly_maps = (recon_dist + seg_dist) / 2 
        anomaly_maps = self.min_max_scaling(anomaly_maps)

        # print(f"{anomaly_maps.shape=}{anomaly_maps.min()=}{anomaly_maps.max()=}")

        pred_scores = self.compute_metrics(
                        outputs["pred_imgs"], outputs["image"], # image,# 
                        outputs['pred_masks'], outputs['gen_masks'], 
                        anomaly_maps, 
                        mode=None,
        )

        return anomaly_maps, pred_scores 

    @staticmethod
    def compute_segmentation_distance(
        masks, 
        preds,targets,
        mode = "cosine", 
        scale = 4, # 64 -> 256
        patch_size = (4,4), # 256 -> 64
        eps = 1e-100, # torch.finfo.eps,
    ):
        assert preds.shape == targets.shape,\
            f"{preds.shape=}\t{targets.shape=}"

        diffs=[]
        for mask, pred, target in zip(masks, preds, targets):
            num_classes = mask.flatten().unique().size(dim=0) + 1
            mask_one_hot = F.one_hot(mask.unsqueeze(0), num_classes=num_classes)
            mask_one_hot = mask_one_hot.permute(0, 3, 1, 2).squeeze(0).to(torch.float)# Convert from NHWC to CHW

            # print(f"{mask_one_hot.shape=}\t{pred.shape=}\t{target.shape=}") # [5, 256, 256] [64, 64, 64] [64, 64, 64]
            pred = F.interpolate(pred.unsqueeze(0), scale_factor=scale, mode="bilinear").squeeze(0) # 64 -> 256
            target = F.interpolate(target.unsqueeze(0), scale_factor=scale, mode="bilinear").squeeze(0) # 64 -> 256

            # クラスごとにmaskがTrueな箇所は残す，それ以外は０
            onehot_pred = []
            for M in mask_one_hot:
                M = M.repeat(pred.shape[0], 1,1)
                _pred = torch.bmm(M, pred.permute(0,2,1,)) # C H W * C W H -> C H W
                onehot_pred.append(_pred)
            onehot_pred = torch.stack(onehot_pred, dim=0)
            onehot_target = []
            for M in mask_one_hot:
                M = M.repeat(target.shape[0], 1,1)
                _target = torch.bmm(M, target.permute(0,2,1,)) # C H W * C W H -> C H W
                onehot_target.append(_target)
            onehot_target = torch.stack(onehot_target, dim=0)
            # print(f"{onehot_pred.shape=}\t{onehot_target.shape=}") # Class, latentChannel, H, W

            # TODO:: patch cosine similarity
#             dist = F.cosine_similarity(
#                 pred.reshape(1, -1), target.reshape(1, -1), 
#                 dim=1, eps=eps,
#             ) 

            onehot_pred = torch.sum(onehot_pred, 0, keepdim=True,).squeeze(0)
            _pred = torch.mean(onehot_pred, 0, keepdim=True,).squeeze(0)
            onehot_target = torch.sum(onehot_target, 0, keepdim=True,).squeeze(0)
            _target = torch.mean(onehot_target, 0, keepdim=True,).squeeze(0)
            diff = torch.sqrt((_pred - _target)**2 + eps) 

            # print(f"{diff.shape=}")
            diffs.append(diff)
        diffs = torch.stack(diffs, dim=0) # B, 1, H, W
        # print(f"{diffs.shape=}")

        return diffs 

    @staticmethod
    def compute_recon_distance(
        pred, target,
        mode = "L2", 
        eps = 1e-100,# torch.finfo.eps,
    ):

        assert pred.shape == target.shape,\
            f"{pred.shape=}\t{target.shape=}"

        diffs=0
        if mode=="L1":
           # https://note.nkmk.me/python-opencv-numpy-image-difference/
           diffs = torch.abs(pred - target) 
        elif mode == "L2":
            diffs = torch.sqrt((pred - target)**2 + eps) 
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
        elif mode == "ms_ssim":
            for pred, img in zip(pred_imgs, imgs):
                score = ms_ssim(pred, img, data_range=1.0)
                scores.append(score)
        else:
            for ano_map in anomaly_maps:
                score = torch.max(ano_map)
                scores.append(score)
        scores = torch.stack(scores, dim=0)
        # print(f"recon  value : {scores},\n\t{scores.shape=}{scores.device=}")

        # Calc segmentation metrics
        _scores = []
        for pred, gen in zip(pred_masks, gen_masks):
            mIoU = self.calc_meanIoU(pred, gen,)
            score = 1 - mIoU.mean()
            _scores.append(score)
        _scores = torch.stack(_scores, dim=0)
        # print(f"segment value : {_scores},\n\t{_scores.shape=}{_scores.device=}")
 
        scores = (scores + _scores)/2

        return scores

    def min_max_scaling(self,v, new_min=0.0, new_max=1.0):
        v_min, v_max = v.min(), v.max()
        v_p = (v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        return v_p

    def cvt_channel_gray2RGB(self,x):
        return x.repeat(1, 3, 1, 1)

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
