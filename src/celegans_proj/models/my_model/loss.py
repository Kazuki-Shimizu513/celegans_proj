

import torch 
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure  as ms_ssim,
    learned_perceptual_image_patch_similarity as lpips,
)
from PIL import Image

from .torch_model import MyTorchModel
# from torch_model import MyTorchModel

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

        loss = self.loss_image_weight * self.compute_recon_loss(outputs['pred_imgs'], outputs['imgs'], perceptual="ssim")

        if "vae" in self.train_models \
            and "diffsion" in self.train_models:
            loss += self.loss_latent_weight * self.compute_recon_loss(outputs['pred_latents'], outputs['latents'])
            loss += self.loss_noise_weight * self.compute_recon_loss(outputs['pred_noises'], outputs['noises'], )
            loss += self.loss_gen_weight * self.compute_recon_loss(outputs['gen_imgs'], outputs['imgs'], )
            loss += self.loss_gen_weight * self.compute_recon_loss(outputs['gen_latents'], outputs['latents'], )

            loss += self.loss_emb_weight * self.compute_embedding_loss(outputs['gen_latents'], outputs['latents'],)
            loss += self.loss_emb_weight * self.compute_embedding_loss(outputs['pred_latents'], outputs['latents'],)

            loss += self.loss_mask_weight * self.compute_recon_loss(outputs['KL_THRESHOLDS'], outputs['gen_KL_THRESHOLDS'], )
            loss += self.loss_mask_weight * self.compute_segmentation_loss(outputs['pred_masks'], outputs['gen_masks'], )

            loss += self.loss_kl_weight * outputs['posterior'].kl().mean()# kl loss

        elif "diffsion" in self.train_models:
            loss += self.loss_latent_weight * self.compute_recon_loss(outputs['pred_latents'], outputs['latents'])
            loss += self.loss_noise_weight * self.compute_recon_loss(outputs['pred_noises'], outputs['noises'], )
            loss += self.loss_gen_weight * self.compute_recon_loss(outputs['gen_imgs'], outputs['imgs'], )
            loss += self.loss_gen_weight * self.compute_recon_loss(outputs['gen_latents'], outputs['latents'], )

            loss += self.loss_emb_weight * self.compute_embedding_loss(outputs['gen_latents'], outputs['latents'],)
            loss += self.loss_emb_weight * self.compute_embedding_loss(outputs['pred_latents'], outputs['latents'],)


            loss += self.loss_mask_weight * self.compute_recon_loss(outputs['KL_THRESHOLDS'], outputs['gen_KL_THRESHOLDS'], )
            loss += self.loss_mask_weight * self.compute_segmentation_loss(outputs['pred_masks'], outputs['gen_masks'], )

        elif "vae" in self.train_models:
            loss += self.loss_kl_weight * outputs['posterior'].kl().mean()# kl loss

        if torch.isnan(loss):
            loss = torch.tensor(0.0, dtype=torch.float32,device=outputs['imgs'].device,requires_grad=True)
            print(f"nan in loss")

        return loss

    @staticmethod
    def compute_embedding_loss(
        pred, gen,
    ):
        assert pred.shape == gen.shape,\
            f"{pred.shape=}\t{gen.shape=}"

        bsz = pred.shape[0]
        pred = pred.reshape(bsz, -1)
        gen = gen.reshape(bsz, -1)
        target = torch.ones(pred.shape[0])

        loss = F.cosine_embedding_loss(
            pred, gen, target, 
            margin=0, size_average=None, reduce=None, reduction='mean'
        ) 
        return loss


    @staticmethod
    def compute_segmentation_loss(
        pred, gen,
    ):

        assert pred.shape == gen.shape,\
            f"{pred.shape=}\t{gen.shape=}"

        sup_loss = 0
        sup_ce = self.ce_loss(pred,gen)
        sup_focal = self.focal_loss(pred,gen)
        sup_dice = self.dice_loss(pred,gen)
        # sup_hist = self.hist_loss(pred,rand_gt)
        sup_entro = self.entropy_loss(pred)
        sup_loss = sup_ce + sup_focal + sup_dice + sup_entro#  + sup_hist

        unsup_loss = 0
        # unsup_hist = self.hist_loss(gen,rand_gt)
        unsup_entro = self.entropy_loss(gen)
        unsup_loss = unsup_entro#  + unsup_hist

        loss = sup_loss + unsup_loss
        return loss


    @staticmethod
    def compute_recon_loss(
        pred, target,
        perceptual = None, 
    ):

        assert pred.shape == target.shape,\
            f"{pred.shape=}\t{target.shape=}"

        loss = F.mse_loss(pred, target)

        if perceptual=="ssim":
            loss += 1 - ms_ssim(pred, target, data_range=1.0)
        elif perceptual=="lpips":
            if pred.shape[1] == 1:
                pred = MyLoss.cvt_channel_gray2RGB(pred)
                target = MyLoss.cvt_channel_gray2RGB(target)

            if pred.min() < 0.0 or target.min() < 0.0:
                pred  = MyLoss.min_max_saling(pred)
                target  = MyLoss.min_max_saling(target)

            loss += 1 - lpips(pred, target, 
                              net_type='squeeze', normalize=True,
                              )

        return loss

    @staticmethod
    def cvt_channel_gray2RGB(x):
        return x.repeat(1, 3, 1, 1)

    @staticmethod
    def min_max_scaling(x, eps=1e-10):
        new_x = (x-x.min())/ (x.max() - x.min() + eps)
        return new_x



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
    training = True # False # True # 
    B = 1
    C,W,H = (1,256,256)
    cuda_0 = torch.device("cuda:0")
    img = torch.randn(B,C,W,H,).to(device=cuda_0)

    img_path = "/mnt/d/WDDD2/TIF_GRAY/wt_N2_081113_01/wt_N2_081113_01_T60_Z49.tiff"
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

    print(f"{img.shape=}")

    model = MyTorchModel(
            training = training,
            train_models=["vae", "diffusion"],
    ).to(device=cuda_0)

    output = model(img)
    print(f"{output.keys()=}")

    loss_fn = MyLoss(
        train_models= ["vae", "diffusion", ], 

        loss_image_weight = 1.0,
        loss_latent_weight = 1.0,
        loss_noise_weight = 1.0,
        loss_kl_weight = 1.0,
        loss_gen_weight = 1.0,
        loss_mask_weight = 1.0,

    ) 
    loss = loss_fn(output)
    loss.backward()
    print(loss)

