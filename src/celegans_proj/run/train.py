
import torch 
from torchvision.transforms.v2 import Transform
from torchvision.transforms import v2

from pathlib import Path
import json

import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)

from anomalib import TaskType
from anomalib.models import (
    Patchcore,
    ReverseDistillation,
)
from anomalib.engine import Engine
from anomalib.data.utils import TestSplitMode

import wandb

from celegans_proj.run.image.wddd2 import (
    WDDD2_AD,
    YouTransform,
) 

from celegans_proj.models import (
    SimSID,
)

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")
torch._dynamo.config.suppress_errors = True

# wandb.login()

def train(
    exp_name = "exp_20241026",
    out_dir = "/mnt/c/Users/compbio/Desktop/shimizudata/",
    in_dir = "/mnt/e/WDDD2_AD",
    target_data = "wildType",
    model_name = "Patchcore",
    ckpt = None, 
    batch = 1,
    resolution = 256,
    task = TaskType.SEGMENTATION, #CLASSIFICATION,#
    threshold =  "F1AdaptiveThreshold",
    worker = 30,
    logger = logger, 
    metric_name = "pixel_AUROC",
):

    out_dir = Path(out_dir)
    out_dir = out_dir.joinpath(f"{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir.joinpath("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor=metric_name, 
        mode='max',
        dirpath = str(model_dir),
        filename='{epoch}-{pixel_AUROC:.2f}',
        save_top_k= 1,
    )

    callbacks = [
        checkpoint_callback,
        EarlyStopping(metric_name),
        DeviceStatsMonitor(),
    ]

    print("prepareing datamodule")
    # https://pytorch.org/vision/main/transforms.html#v2-api-ref
    transforms = v2.Compose([
                    v2.Grayscale(), # Wide resNet has 3 channels
                    v2.PILToTensor(),
                    v2.Resize(
                        size=(resolution, resolution), 
                        interpolation=v2.InterpolationMode.BILINEAR
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    # TODO: YouTransform
                    v2.Normalize(mean=[0.485], std=[0.229]),
                    # v2.Normalize(
                    #     mean=[0.485, 0.456, 0.406], 
                    #     std=[0.229, 0.224, 0.225],
                    # ),
                ]) 

    # Initialize the datamodule, model and engine
    datamodule = WDDD2_AD(
        root = in_dir ,
        category = target_data,
        train_batch_size = batch,
        eval_batch_size = batch,
        test_split_mode=TestSplitMode.SYNTHETIC,
        num_workers = worker, 
        task = TaskType.SEGMENTATION,#CLASSIFICATION,#  
        image_size = (resolution,resolution),
        transform = transforms,
        seed  = 44,
    )

    print("prepareing model")

    if model_name == "Patchcore":
        model = Patchcore(
            backbone = "wide_resnet50_2",
            layers = ("layer2", "layer3"),
            pre_trained = True,
            coreset_sampling_ratio = 0.1,
            num_neighbors = 9,
        )
    elif model_name == "ReverseDistillation":
        model = ReverseDistillation(
            backbone = "wide_resnet50_2",
            layers = ("layer1", "layer2", "layer3"),
            pre_trained = True,
        )
    elif model_name == "SimSID":

        model = SimSID()
    else:
        logger.info("please give model_name Patchcore or ReverseDistillation or SimSID")
        return 0

    model = torch.compile(model)

    print("prepareing Trainer")
    engine = Engine(
        logger = logger,
        callbacks = callbacks,
        default_root_dir= str(out_dir),
        threshold = threshold,
        max_epochs = -1,
    )

    print("starting Training")
    engine.fit(
        datamodule=datamodule,
        model=model,
        ckpt_path = ckpt,
    )


if __name__ == "__main__":

    logging.basicConfig(filename='./logs/debug.log', filemode='w', level=logging.DEBUG)

    # exp_name  = "exp_example"
    exp_name  = "exp_20241109"

    out_dir = "/mnt/c/Users/compbio/Desktop/shimizudata/"
    log_dir  = "./logs"
    in_dir = "/mnt/e/WDDD2_AD"
    model_name = "SimSID"
    target_data = "wildType"
    threshold =  "F1AdaptiveThreshold"
    metric_name = "pixel_AUROC"


    logger = WandbLogger(
        project =f"{exp_name}",
        name = f"{model_name}_{target_data}",
        save_dir = str(log_dir),
        log_model = "all",#  True, 
    )
    # logger.experiment.config.update(**vars(args))


    train(
        exp_name = exp_name,
        out_dir = out_dir,
        in_dir = in_dir,
        model_name = model_name,
        target_data = target_data,
        threshold =  threshold,
        logger = logger, 
        metric_name = metric_name,

        ckpt = None, 
        batch = 1,
        resolution = 128, # 256,
        task = TaskType.SEGMENTATION, #CLASSIFICATION,#
        worker = 30,
    )

