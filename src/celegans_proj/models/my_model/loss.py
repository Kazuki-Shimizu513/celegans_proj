
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchmetrics.functional.image import (
    peak_signal_noise_ratio as PSNR,
    multiscale_structural_similarity_index_measure  as ms_ssim,
    learned_perceptual_image_patch_similarity as lpips,
)
import numpy as np
from PIL import Image

from celegans_proj.models.my_model.torch_model import MyTorchModel

class MyLoss(nn.Module):
    """Loss function """
    def __init__(
        self,
        train_models= ["vae", "diffusion", ], 

        loss_image_weight = 1.0,
        loss_latent_weight = 1.0,
        loss_noise_weight = 1.0,
        loss_kl_weight = 1.0,
        loss_gen_weight = 1.0,
        loss_emb_weight = 1.0,
        loss_mask_weight = 1.0,

    ) -> None:
        super().__init__()
        self.train_models = train_models

        self.loss_image_weight = loss_image_weight
        self.loss_latent_weight = loss_latent_weight
        self.loss_noise_weight = loss_noise_weight
        self.loss_kl_weight = loss_kl_weight
        self.loss_gen_weight = loss_gen_weight
        self.loss_emb_weight = loss_emb_weight
        self.loss_mask_weight = loss_mask_weight


        self.ce_loss = MulticlassCrossEntropyLoss(ignore_index=None)
        self.focal_loss = FocalLoss(ignore_index=None)
        self.dice_loss = ClassBalancedDiceLoss(ignore_index=None)
        self.hist_loss = HistLoss(ignore_index=0)
        self.entropy_loss = EntropyLoss()

    def forward(
        self,
        outputs, 
    ) -> torch.Tensor:
        """

        Args:

        Returns:
            Tensor: loss
        """

        loss = self.loss_image_weight * self.compute_recon_loss(
            outputs['pred_imgs'], outputs['image'], 
            perceptual="ms_ssim",# "lpips",
        )

        # loss += self.loss_image_weight * self.ce_loss(
        #     outputs['anomaly_maps'],   
        #     outputs['mask'].to(torch.long),
        #  )

        loss += self.loss_image_weight * self.compute_recon_loss(
            outputs['pred_scores'], 
            outputs['label'],
         )

        if "vae" in self.train_models \
            and "diffusion" in self.train_models:

            loss += self.loss_latent_weight * self.compute_recon_loss(
                outputs['pred_latents'], outputs['latents'])
            loss += self.loss_noise_weight * self.compute_recon_loss(
                outputs['pred_noises'], outputs['noises'], )

            # loss += self.loss_gen_weight * self.compute_recon_loss(
            #     outputs['gen_imgs'], outputs['image'], )
            # loss += self.loss_gen_weight * self.compute_recon_loss(
            #     outputs['gen_latents'], outputs['latents'], )

            # loss += self.loss_emb_weight * self.compute_embedding_loss(
            #     outputs['gen_latents'], outputs['latents'],)
            loss += self.loss_emb_weight * self.compute_embedding_loss(
                outputs['pred_latents'], outputs['latents'],)

            # loss += self.loss_mask_weight * self.compute_divergence_loss(
            #     outputs['KL_THRESHOLDS'], outputs['gen_KL_THRESHOLDS'], )

            # loss += self.loss_kl_weight * outputs['posterior'].kl().mean()# kl loss

            # loss += self.loss_mask_weight * self.compute_segmentation_loss(
            #     outputs['pred_masks'], outputs['seg_masks'], )
            # loss += self.loss_mask_weight * self.compute_recon_loss(
            #     outputs['pred_masks'], outputs['gen_masks'], )

        elif "diffusion" in self.train_models:
            loss += self.loss_latent_weight * self.compute_recon_loss(
                outputs['pred_latents'], outputs['latents'])
            loss += self.loss_noise_weight * self.compute_recon_loss(
                outputs['pred_noises'], outputs['noises'], )

            # loss += self.loss_gen_weight * self.compute_recon_loss(
            #     outputs['gen_imgs'], outputs['image'], )
            # loss += self.loss_gen_weight * self.compute_recon_loss(
            #     outputs['gen_latents'], outputs['latents'], )

            loss += self.loss_emb_weight * self.compute_embedding_loss(
                outputs['pred_latents'], outputs['latents'],)
            # loss += self.loss_emb_weight * self.compute_embedding_loss(
            #     outputs['gen_latents'], outputs['latents'],)



            # loss += self.loss_mask_weight * self.compute_divergence_loss(
            #     outputs['KL_THRESHOLDS'], outputs['gen_KL_THRESHOLDS'], )
            # loss += self.loss_mask_weight * self.compute_segmentation_loss(
                # outputs['pred_masks'], outputs['seg_masks'], )
            # loss += self.loss_mask_weight * self.compute_recon_loss(outputs['pred_masks'], outputs['gen_masks'], )

        elif "vae" in self.train_models:
            pass
            # loss += self.loss_kl_weight * outputs['posterior'].kl().mean()# kl loss

        if torch.isnan(loss):
            loss = torch.tensor(1.0, dtype=torch.float32,device=outputs['image'].device,requires_grad=True)
            print(f"nan in loss")

        if loss.dtype != torch.float:
            try:
                loss = loss.to(torch.float)
            except:
                loss = torch.tensor(1.0, dtype=torch.float32,device=outputs['image'].device,requires_grad=True)
            print(f"{loss.dytpe} in loss")

        return loss

    @staticmethod
    def compute_divergence_loss(
            p,q, 
    ):
        p, q = p.view(-1, p.size(-1)),q.view(-1, q.size(-1)), # -> size([1,4])
        m = (0.5 * (p + q))
        p, q, m = F.log_softmax(p,dim=1),F.log_softmax(q, dim=1),F.log_softmax(m, dim=1)
        kl_mp = F.kl_div(m, p, log_target=True,reduction="batchmean",)
        kl_mq = F.kl_div(m, q, log_target=True,reduction="batchmean",)
        js_div = 0.5 * (kl_mq + kl_mp)
        return js_div

    @staticmethod
    def compute_embedding_loss(
        pred, gen,
    ):
        assert pred.shape == gen.shape,\
            f"{pred.shape=}\t{gen.shape=}"

        bsz = pred.shape[0]
        pred = pred.reshape(bsz, -1)
        gen = gen.reshape(bsz, -1)
        target = torch.ones(pred.shape[0], device=pred.device)

        loss = F.cosine_embedding_loss(
            pred, gen, target, 
            margin=0, size_average=None, reduce=None, reduction='mean'
        ) 
        return loss


    def compute_segmentation_loss(
        self,
        pred, target,
    ):

        assert pred.shape == target.shape,\
            f"{pred.shape=}\t{target.shape=}"
        classes = self.get_segClasses(pred, target)
        pred = self.fmt_segMask2entropy(pred, num_class = classes)
        target_entro = self.fmt_segMask2entropy(target, num_class = classes)

        sup_loss = 0
        sup_ce = self.ce_loss(pred,target)
        sup_focal = self.focal_loss(pred,target)
        sup_dice = self.dice_loss(pred,target)
        # sup_hist = self.hist_loss(pred,rand_gt)
        # sup_entro = self.entropy_loss(pred) + self.entropy_loss(target)
        sup_loss = sup_ce + sup_focal + sup_dice #+ sup_entro + sup_hist

        unsup_loss = 0
        # unsup_hist = self.hist_loss(gen,rand_gt)
        unsup_entro = self.entropy_loss(pred) + self.entropy_loss(target_entro)
        unsup_loss = unsup_entro #+ unsup_hist

        loss = sup_loss + unsup_loss

        # print(f"In segmentation {loss=}\t{loss.shape=}")
        return loss

    @staticmethod
    def get_segClasses(msk1, msk2):

        cls1 = msk1.unique().tolist()
        cls2 = msk2.unique().tolist()
        classes = max([*cls1, *cls2]) + 1

        return classes 

    @staticmethod
    def fmt_segMask2entropy(mask, num_class=5, eps=1e-100):

        mask_one_hot = F.one_hot(mask.squeeze(1), num_classes=num_class)
        # Convert from NHWC to NCHW
        mask_one_hot = mask_one_hot.permute(0, 3, 1, 2).type(torch.float)

        probabilities = nn.Softmax(dim=1)(mask_one_hot)
        probabilities -= eps # 1e-10

        return probabilities 

    @staticmethod
    def compute_recon_loss(
        pred, target,
        perceptual = None, 
    ):

        assert pred.shape == target.shape,\
            f"{pred.shape=}\t{target.shape=}"

        if pred.dtype == torch.int64:
            pred = pred.to(torch.float)
        if target.dtype == torch.int64:
            target = target.to(torch.float)

        loss = F.mse_loss(pred, target , reduction="sum",) #"mean",)# 

        if perceptual=="ssim":
            loss += 1 - ms_ssim(pred, target, data_range=1.0, reduction="sum",) # "mean"

        elif perceptual=="lpips":
            if pred.shape[1] == 1:
                pred = MyLoss.cvt_channel_gray2RGB(pred)
                target = MyLoss.cvt_channel_gray2RGB(target)

            loss += 1 - lpips(pred, target, 
                              net_type='squeeze', 
                              normalize=True,
                              reduction="sum", # "mean"
                              )

        return loss

    @staticmethod
    def cvt_channel_gray2RGB(x):
        return x.repeat(1, 3, 1, 1)

    @staticmethod
    def min_max_scaling(x, feature_range=(0,1), eps=1e-100,):#torch.finfo(torch.float32).eps
        x_std = (x - x.min())/ (x.max() - x.min() + eps)
        new_x = x_std * (feature_range[0] - feature_range[1]) + feature_range[0]
        return new_x




# https://github.com/Tokichan/CSAD/blob/main/models/segmentation/loss.py
class MulticlassCrossEntropyLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probabilities = logits

        probabilities = nn.Softmax(dim=1)(logits)
        # end if
        targets_one_hot = torch.nn.functional.one_hot(targets.squeeze(1), num_classes=logits.shape[1])
        # print(targets_one_hot.shape)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)

        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)
        
        return self.ce_loss(probabilities,targets_one_hot)
    
class MulticlassDiceLoss(nn.Module):
    """Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets, reduction='mean', smooth=1e-6):
        """The "reduction" argument is ignored. This method computes the dice
        loss for all classes and provides an overall weighted loss.
        """
        probabilities = logits

        probabilities = nn.Softmax(dim=1)(logits)
        # end if
        targets_one_hot = torch.nn.functional.one_hot(targets.squeeze(1), num_classes=logits.shape[1])
        # print(targets_one_hot.shape)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)

        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)
        
        # Multiply one-hot encoded ground truth labels with the probabilities to get the
        # prredicted probability for the actual class.
        intersection = (targets_one_hot * probabilities).sum()
        
        mod_a = intersection.sum()
        mod_b = targets.numel()
        
        dice_coefficient = 2. * intersection / (mod_a + mod_b + smooth)
        dice_loss = -dice_coefficient.log()
        return dice_loss
    
class ClassBalancedDiceLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        probabilities = torch.softmax(prediction,dim=1)
        targets_one_hot = torch.nn.functional.one_hot(target.squeeze(1), num_classes=prediction.shape[1])
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)

        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)


        class_weights = self._calculate_class_weights(targets_one_hot)
        dice_loss = self._dice_loss(probabilities, targets_one_hot)
        class_balanced_loss = class_weights * dice_loss
        return class_balanced_loss.mean()

    def _calculate_class_weights(self, target):
        """
        Calculates class weights based on their inverse frequency in the target.
        """
        weights = torch.zeros((target.shape[0],target.shape[1])).cuda()
        for c in range(target.shape[1]):
            weights[:,c] = 1 / (target[:,c].sum() + 1e-5)
        weights = weights / weights.sum(dim=1,keepdim=True)
        return weights.detach()

    def _dice_loss(self, prediction, target):
        """
        Calculates dice loss for each class and then averages across all classes.
        """
        intersection = 2 * (prediction * target).sum(dim=(2, 3))
        union = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-5
        dice = (intersection + 1e-5) / (union + 1e-5)
        return 1 - dice
    

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, ignore_index=None, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        self.ignore_index = ignore_index

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        logit = torch.softmax(logit, dim=1)
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.ignore_index is not None:
            one_hot_key = torch.concat([one_hot_key[:,:self.ignore_index],one_hot_key[:,self.ignore_index+1:]],dim=1)
            logit = torch.concat([logit[:,:self.ignore_index],logit[:,self.ignore_index+1:]],dim=1)


        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss
    

class OldHistLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits,targets):
        # ignore background
        probabilities = logits

        probabilities = nn.Softmax(dim=1)(probabilities)

        targets_one_hot = torch.nn.functional.one_hot(targets.squeeze(1), num_classes=logits.shape[1])

        # print(targets_one_hot.shape)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)
    
        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)
    
        
        targets_hist = torch.mean(targets_one_hot,dim=(2,3)) # (B,C)
        targets_hist = torch.nn.functional.normalize(targets_hist,dim=1)
        
        preds_hist = torch.mean(probabilities,dim=(2,3)) # (B,C)
        preds_hist = torch.nn.functional.normalize(preds_hist,dim=1)

        hist_loss = torch.mean(torch.abs(targets_hist-preds_hist))
        return hist_loss
    
class HistLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self,pred, trg):
        pred = torch.softmax(pred,dim=1)
        new_trg = torch.zeros_like(trg).repeat(1, pred.shape[1], 1, 1).long()
        new_trg = new_trg.scatter(1, trg, 1).float()
        diff = torch.abs(new_trg.mean((2, 3)) - pred.mean((2, 3)))
        if self.ignore_index is not None:
            diff = torch.concat([diff[:,:self.ignore_index],diff[:,self.ignore_index+1:]],dim=1)
        loss = diff.sum() / pred.shape[0]  # exclude BG
        return loss
    
class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred):
        prob = nn.Softmax(dim=1)(pred)
        return (-1*prob*((prob+1e-5).log())).mean()






if __name__ == "__main__":
    # __debug__ = True # when you execute without any option 

    # image_path      ['/mnt/e/WDDD2_AD/wildType/train/good/wt_N2_081105_03_T28_Z34.tiff']
    # label   tensor([0], device='cuda:0')
    # image   tensor([[[[-1.3911, -1.3935, -1.3924,  ..., -1.3850, -1.3887, -1.3922],
    #                   [-1.3971, -1.3901, -1.3910,  ..., -1.3846, -1.3820, -1.3861],
    #                   [-1.3799, -1.3889, -1.3901,  ..., -1.3844, -1.3852, -1.3824],
    #                   ...,
    #                   [-1.3875, -1.3845, -1.3844,  ..., -1.4095, -1.4101, -1.4099],
    #                   [-1.3790, -1.3942, -1.3882,  ..., -1.4054, -1.4141, -1.4114],
    #                   [-1.3888, -1.3918, -1.3906,  ..., -1.4129, -1.4155, -1.4158]]]],device='cuda:0')
    # mask    tensor([[[0, 0, 0,  ..., 0, 0, 0],
    #                  [0, 0, 0,  ..., 0, 0, 0],
    #                  [0, 0, 0,  ..., 0, 0, 0],
    #                  ...,
    #                  [0, 0, 0,  ..., 0, 0, 0],
    #                  [0, 0, 0,  ..., 0, 0, 0],
    #                  [0, 0, 0,  ..., 0, 0, 0]]], device='cuda:0', dtype=torch.uint8)

    cuda_0 = torch.device("cuda:0")
    training = True # False # True # 
    training_mask = True

    B = 1
    C,W,H = (1,256,256)

    img_path = "/mnt/d/WDDD2/TIF_GRAY/wt_N2_081113_01/wt_N2_081113_01_T60_Z49.tiff"
    img = torch.randn(B,C,W,H,).to(device=cuda_0)
    img = Image.open(img_path)
    transforms = v2.Compose([
                    v2.Grayscale(), # Wide resNet has 3 channels
                    v2.PILToTensor(),
                    v2.Resize(
                        size=(W, H), 
                        interpolation=v2.InterpolationMode.BILINEAR
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    # TODO: YouTransform
                    v2.Normalize(mean=[0.485], std=[0.229]),
                ]) 
    img = transforms(img)
    img = img.to(device=cuda_0)
    img = img.unsqueeze(0)
    img = img.repeat(B, 1, 1, 1)# .permute(1, 2, 0) # C*3, W, H -> W, H, C*3,
    print(f"{img.shape=}")
    msk = torch.zeros(B,W,H,).to(device=cuda_0)
    lbl = torch.tensor([np.int64(0)]* B).to(device=cuda_0)


    batch = dict(
        image_path=[img_path]* B,
        image=img,
        mask=msk,
        label=lbl,
    )

    model = MyTorchModel(
            training = training,
            training_mask = training_mask,
            train_models=["vae", "diffusion"],
    ).to(device=cuda_0)

    outputs = model(batch)
    print(f"{outputs.keys()=}")

    loss_fn = MyLoss(
        train_models= ["vae", "diffusion", ], 

        loss_image_weight = 1.0,
        loss_latent_weight = 1.0,
        loss_noise_weight = 1.0,
        loss_kl_weight = 1.0,
        loss_gen_weight = 1.0,
        loss_mask_weight = 1.0,
    ) 

    loss = loss_fn(outputs)
    loss.backward()
    print(loss)

