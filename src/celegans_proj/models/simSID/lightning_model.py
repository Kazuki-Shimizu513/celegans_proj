
from collections.abc import Sequence
from typing import Any


import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import LearningType
from anomalib.models.components import AnomalyModule


from .components import (
    MemoryQueue,
    # GanAlert,
) 
from .wddd2_ad_config import Config 
from .torch_model import (
    SQUID, 
    SimpleDiscriminator,
)



class SimSID(AnomalyModule):
    """PL Lightning Module for SimSID Algorithm.

    Args:
        backbone (str): Backbone of CNN network
            Defaults to ``wide_resnet50_2``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.


    """
    def __init__(
        self, 
    ) -> None:
        super().__init__()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self.CONFIG = Config()

        self.model : SQUID
        self.discriminator : SimpleDiscriminator 

        self.ce = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.MSELoss(reduction='mean')

        # for sub GAN discriminator# alert
        # self.alert = GanAlert(
        #     generator=self.model,
        #     discriminator=self.discriminator, 
        #     CONFIG=self.CONFIG, 
        # )

    def _setup(
        self, 
    ) -> None:
        self.model = SQUID(
            self.CONFIG, 32, level=self.CONFIG.level
        )

        self.discriminator = SimpleDiscriminator(
            self.CONFIG.num_in_ch, 
            size=self.CONFIG.size, 
            num_layers=self.CONFIG.num_layers
        )

    def configure_optimizers(
        self, 
    ) -> None:

        # for main AE
        opt = optim.Adam(
            self.model.parameters(), 
            lr=self.CONFIG.lr, 
            eps=1e-7, 
            betas=(0.5, 0.999), 
            weight_decay=0.00001
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=300, 
            # eta_min=self.lr*0.5,
        )
        opt_settings = [
            {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                }
            },
        ]

        # for sub GAN discriminator
        opt_d = optim.Adam(
            self.discriminator.parameters(), 
            lr=self.CONFIG.gan_lr,
            betas=(0.5, 0.999), 
        )
        # scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
        #     opt_d, 
        #     T_max=200, 
        #     eta_min=self.lr*0.2,
        # )
        opt_settings.append(
            {
                "optimizer": opt_d,
                # "lr_scheduler": {
                #     "scheduler": scheduler_d,
                #     'monitor': 'val_loss',
                #     'interval': 'epoch',
                # }
            },
        )

        return tuple(opt_settings)

    def training_step(
        self, 
        batch: dict[str, str | torch.Tensor], 
        batch_idx,
        *args, **kwargs
    ) -> None:
        """Perform a training step of SimSID Model.

        Args:
          batch (dict[str, str | torch.Tensor]): Input batch
          args: Additional arguments.
          kwargs: Additional keyword arguments.

        Returns:
          loss (torch.Tensor): 

        References:
            -  https://lightning.ai/docs/pytorch/stable/model/build_model_advanced.html#use-multiple-optimizers-like-gans
        """
        if batch_idx > self.CONFIG.limit:
            return None

        img = batch["image"]
        out = self.model(img, fadein_weights= [0.0, 0.0, 1.0, 1.0])
        opt, opt_d = self.optimizers()

        ##########################
        # Optimize Discriminator #
        ##########################

        d_loss, real_validity, fake_validity = self._calc_disc_loss(img, out["recon"], )

        # backward discriminator loss
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        ######################
        # Optimize Generator #
        ######################

        # train generator at every n_critic step only
        if batch_idx % self.CONFIG.n_critic == 0:
            loss, recon_loss, g_loss, t_recon_loss, dist_loss = self._calc_generator_loss(img, out)

            # backward discriminator loss
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

        ###############
        # log metrics #
        ###############
        tot_loss = {}
        tot_loss['train_d_loss'] = d_loss.item()
        tot_loss['train_loss'] = d_loss.item()
        tot_loss['train_normalScore'] = real_validity.detach().cpu().numpy()
        tot_loss['train_anomalyScore'] = fake_validity.detach().cpu().numpy()

        # train generator at every n_critic step only
        if batch_idx % self.CONFIG.n_critic == 0:
            tot_loss['train_recon_loss'] = recon_loss.item()
            tot_loss['train_g_loss'] = g_loss.item()
            tot_loss['train_teacher_recon_loss'] = t_recon_loss.item()
            tot_loss['train_dist_loss'] = dist_loss.item()
            tot_loss['train_loss'] += loss.item()

        self.log_dict(tot_loss, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        for module in self.model.modules():
            if isinstance(module, MemoryQueue):
                module.update()

    def validation_step(
        self, 
        batch,
        batch_index,
        *args, **kwargs,
    ) -> None:
        """

        """
        img = batch["image"]
        label = batch["label"]

        out = self.model(img, fadein_weights= [0.0, 0.0, 1.0, 1.0], )
        diff_l1 = self.calc_difference(out['recon'] - img)
        batch["anomaly_map"] = diff_l1

        d_loss, real_v, fake_v = self._calc_disc_loss(img, out["recon"], )
        loss, recon_loss, g_loss, t_recon_loss, dist_loss = self._calc_generator_loss(img, out)

        # TODO:: calc_metric
        # results = self.GanAlert.alert(
        #     fake_v, label, 
        #     mean=0, std=1., print_info=False
        # )
        # self.log_dict(results, prog_bar=True)

        tot_loss = {}
        tot_loss['val_diff_l1'] = diff_l1

        tot_loss['val_normalScore'] = real_v.detach().cpu().numpy()
        tot_loss['val_anomalyScore'] = fake_v.detach().cpu().numpy()

        tot_loss['val_d_loss'] = d_loss.item()
        tot_loss['val_loss'] = d_loss.item()
        tot_loss['val_recon_loss'] = recon_loss.item()
        tot_loss['val_g_loss'] = g_loss.item()
        tot_loss['val_teacher_recon_loss'] = t_recon_loss.item()
        tot_loss['val_dist_loss'] = dist_loss.item()
        tot_loss['val_loss'] += loss.item()

        self.log_dict(tot_loss, prog_bar=True)


        return batch

    def _calc_disc_loss(self,img, recon, ):

        real_validity = self.discriminator(img)
        fake_validity = self.discriminator(recon.detach())

        # discriminator loss
        d_loss = self.ce(real_validity, torch.ones_like(real_validity))
        d_loss += self.ce(fake_validity, torch.zeros_like(fake_validity))
        d_loss *= self.CONFIG.d_w # discriminator loss weight

        return d_loss, real_validity, fake_validity

    def _calc_generator_loss(self,img, out):

        # reconstruction loss
        recon_loss =  self.recon_criterion(out["recon"], img)
        recon_loss *= self.CONFIG.recon_w 

        # generator loss
        fake_validity = self.discriminator(out["recon"])
        g_loss = self.ce(fake_validity, torch.ones_like(fake_validity))
        g_loss *= self.CONFIG.g_w # generator loss weight

        # teacher decoder loss
        if  self.CONFIG.dist \
            and 'teacher_recon' in out \
            and torch.is_tensor(out['teacher_recon']):
            t_recon_loss =  self.recon_criterion(out["teacher_recon"], img)
            t_recon_loss *= self.CONFIG.t_w

        # distillation loss
        if  self.CONFIG.dist \
            and 'dist_loss' in out \
            and torch.is_tensor(out['dist_loss']):
            dist_loss =  out["dist_loss"]
            dist_loss *= self.CONFIG.dist_w  

        loss = recon_loss + g_loss + t_recon_loss + dist_loss

        return loss, recon_loss, g_loss, t_recon_loss, dist_loss 


    def calc_difference(
        self,
        pred,
        target,
        mode = "L1",
        detach = True,
    ):
        if mode == "L1":
            diff = pred - target
            diff = torch.abs(diff)
            diff = torch.mean(diff)

        if detach :
            diff = diff.item()
        return diff


    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return SimSID trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

if __name__ == "__main__":
    pass

