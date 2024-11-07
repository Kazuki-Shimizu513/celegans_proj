
import torch 
from torchvision.transforms.v2 import Transform
from torchvision.transforms import v2

from pathlib import Path

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
from celegans_proj.run.train import train

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")
torch._dynamo.config.suppress_errors = True

# wandb.login()

def predict(
    exp_name = "exp_example",
    out_dir = "/mnt/c/Users/compbio/Desktop/shimizudata/",
    in_dir =  "/mnt/e/WDDD2_AD",
    target_data = "patchBlack",
    model_name = "Patchcore",
    ckpt = "./results/exp_example/Patchcore/WDDD2_AD/wildType/v5/weights/lightning/model.ckpt",
    batch = 1,
    resolution = 256,
    task = TaskType.SEGMENTATION, #CLASSIFICATION,#
    threshold =  "F1AdaptiveThreshold",
    worker = 30,
    logger = logger,
):

    out_dir = Path(out_dir)
    out_dir = out_dir.joinpath(f"{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)


    callbacks = [
        DeviceStatsMonitor()
    ]

    print("prepareing datamodule")
    # https://pytorch.org/vision/main/transforms.html#v2-api-ref
    transforms = v2.Compose([
                    # v2.Grayscale(), # Wide resNet has 3 channels
                    v2.PILToTensor(),
                    v2.Resize(
                        size=(resolution, resolution, ),
                        interpolation=v2.InterpolationMode.BILINEAR,
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    # TODO: YouTransform
                    # v2.Normalize(mean=[0.485], std=[0.229]),
                    v2.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225],
                    ),
                ]) 

    datamodule = WDDD2_AD(
        root = in_dir,
        category = target_data,
        train_batch_size = batch,
        eval_batch_size = batch,
        num_workers = worker, 
        task =  task, 
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
    else:
        logger.info("please give model_name Patchcore or ReverseDistillation")

    model = torch.compile(model)

    print("prepareing Tester")
    engine = Engine(
        logger = logger,
        callbacks = callbacks,
        task = task,
        default_root_dir= str(out_dir),
        threshold = threshold, # "F1AdaptiveThreshold",
    )



    print("starting predictions")
    predictions = engine.predict(
        model = model,
        datamodule = datamodule,
        ckpt_path=ckpt,
    )

    print("saving predictions")
    out_dir = out_dir.joinpath(f"{model_name}/WDDD2_AD/{target_data}")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir.joinpath(f"predictions.pt")
    torch.save(predictions, str(out_file))








if __name__ == "__main__":

    logging.basicConfig(filename='./logs/debug.log', filemode='w', level=logging.DEBUG)

    pseudo_anomaly_modes = [
                            "wildType",
                            "patchBlack",
                            "gridBlack",
                            "zoom",
                            "shrink",
                            "oneCell",
                        ]


    log_dir  = Path("./logs")
    # in_dir = Path("/mnt/e/WDDD2_AD")
    in_dir = Path("/home/skazuki/data/WDDD2_AD")
    # out_dir = Path("/mnt/c/Users/compbio/Desktop/shimizudata/")
    out_dir = Path("/home/skazuki/playground/CaenorhabditisElegans_AnomalyDetection_Dataset/results/")

    model_names = [ "Patchcore",  "ReverseDistillation",]
    # model_names = [ "Patchcore",]
    # model_names = [ "ReverseDistillation",]

    exo_name = "exp_example"
    ckpt_path  = out_dir.joinpath(f"{exp_name}/{model_name}/WDDD2_AD/wildType/v0/weights/lightning/model.ckpt")
    # ckpt_path  = out_dir.joinpath(f"exp_example/Patchcore/WDDD2_AD/wildType/v3/weights/lightning/model.ckpt")
    # ckpt_path  = out_dir.joinpath(f"exp_model/ReverseDistillation/WDDD2_AD/wildType/v0/weights/lightning/model.ckpt")
    for kind in pseudo_anomaly_modes :

        for model_name in  model_names :

            logger = WandbLogger(
                project="exp_example",
                # name = "Patchcore_wildType",
                name = f"{model_name}_{kind}",
                log_model = "all",#  True, 
                save_dir = str(log_dir),
            )


            predict(
                exp_name =f"{exp_name}_1",
                out_dir = out_dir,
                in_dir =  in_dir,
                target_data = kind, # "wildType",
                model_name = model_name, 
                ckpt = ckpt_path,
                batch = 1,
                resolution = 256,
                task = TaskType.SEGMENTATION, #CLASSIFICATION,#
                worker = 16,
                logger = logger,
            )
