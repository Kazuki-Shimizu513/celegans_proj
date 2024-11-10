
from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from anomalib import LearningType
from anomalib.models.components import AnomalyModule

from .torch_model import MyTorchModel


class MyModel(AnomalyModule):
    """PL Lightning Module for My model Algorithm.

    Args:

   """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer1", "layer2", "layer3"),
        pre_trained: bool = True,

    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.pre_trained = pre_trained
        self.layers = layers

        self.model: MyTorchModel
        self.loss = torch.nn.MSELoss()

    def _setup(self) -> None:

        self.model = MyTorchModel(
            backbone=self.backbone,
            pre_trained=self.pre_trained,
            layers=self.layers,
            input_size=self.input_size,
        )

    def configure_optimizers(self) -> optim.Adam:
        """Configure optimizers for decoder and bottleneck.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        return optim.Adam(
            params=list(self.model.decoder.parameters()) + list(self.model.bottleneck.parameters()),
            lr=0.005,
            betas=(0.5, 0.99),
        )

    def training_step(self, batch: dict[str, str | torch.Tensor], batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step 

        Args:
            batch (batch: dict[str, str | torch.Tensor]): Input batch
            batch_idx : Input Batch index
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            loss

        """
        del args, kwargs  # These variables are not used.

        loss = self.loss(*self.model(batch["image"]))
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | torch.Tensor], batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step

        Args:
          batch (dict[str, str | torch.Tensor]): Input batch
            batch_idx : Input Batch index
          args: Additional arguments.
          kwargs: Additional keyword arguments.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """

        batch["anomaly_maps"] = self.model(batch["image"])
        return batch

    def predict_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform a prediction step

        Args:
          batch (dict[str, str | torch.Tensor]): Input batch
            batch_idx : Input Batch index
          args: Additional arguments.
          kwargs: Additional keyword arguments.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `predict_epoch_end` for feature concatenation.
        """

        batch["anomaly_maps"] = self.model(batch["image"])
        return batch


    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS

if __name__ == "__main__":
    print("hello")
