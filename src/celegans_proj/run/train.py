
import torch 
from torchvision.transforms.v2 import Transform
from torchvision.transforms import v2

from pathlib import Path
import json

import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from lightning.pytorch import seed_everything 
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
from anomalib.data.utils import TestSplitMode,ValSplitMode

from anomalib.metrics import ManualThreshold

import wandb

from celegans_proj.run.image import (
    WDDD2_AD,
    YouTransform,
    MyVisualizationCallback,
    MyVisualizer,
    VisualizationMode,
) 

from celegans_proj.models import (
    MyModel,
    SimSID,
)

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_scalar_outputs = True


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
    image_metrics  = ['F1Score'],
    pixel_metrics = ['AUROC'],
    learning_rate  = 1e-8,
    worker = 30,
    logger = logger, 
    seed  =  44,
    debug =  False, 
    debug_data_ratio = 0.01, 
):

    seed_everything(seed)

    out_dir = Path(out_dir)
    out_dir = out_dir.joinpath(f"{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir.joinpath("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir.joinpath("images")
    image_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="pixel_AUROC", # pixel_metrics[0], 
        mode='max',
        dirpath = str(model_dir),
        filename='{epoch}',
        save_top_k= 1,
    )
    
    vis_callback = MyVisualizationCallback(
        visualizers=MyVisualizer(
            mode = VisualizationMode.FULL, # SIMPLE,
            task=task,
            normalize=False
        ),
        save=True,
        root=str(image_dir),
    )

    callbacks = [
        checkpoint_callback,
        EarlyStopping("pixel_AUROC"),
        DeviceStatsMonitor(),
        # vis_callback,
    ]

    print("prepareing datamodule")
    # https://pytorch.org/vision/main/transforms.html#v2-api-ref
    transforms = v2.Compose([
                    # v2.Grayscale(), # Wide resNet has 3 channels
                    v2.PILToTensor(),
                    v2.Resize(
                        size=(resolution, resolution), 
                        interpolation=v2.InterpolationMode.BILINEAR
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    # TODO: YouTransform
                    # v2.Normalize(mean=[0.485], std=[0.229]),
                    v2.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225],
                    ),
                ]) 

    if model_name == "MyModel":# ""SimSID":
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
        num_workers = worker, 
        task = task , # TaskType.SEGMENTATION,#CLASSIFICATION,#  
        val_split_mode = ValSplitMode.FROM_TEST,
        val_split_ratio = 0.01,
        image_size = (resolution,resolution),
        transform = transforms,
        seed  = seed,# 44,
        debug = debug, # True, 
        debug_data_ratio = debug_data_ratio, # 0.01, 
        add_anomalous=debug,
    )
 
    datamodule.setup()

    for attribute in ("train_data", "val_data", "test_data"):
        data_loader = getattr(datamodule,attribute)
        print(f"total {attribute} iterations: {len(data_loader)}")

    i, batch = next(enumerate(datamodule.train_data))
    print(batch.keys()) # dict_keys(['image_path', 'label', 'image', 'mask'])
    print(f"{type(batch['image_path'])=}\t{batch['image_path']=}")
    print(f"{type(batch['label'])=}\t{batch['label']=}")
    print(f"{type(batch['image'])=}\t{batch['image'].shape=}")
    print(f"{type(batch['mask'])=}\t{batch['mask'].shape=}")
    print()

    # return 0
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
    elif model_name == "MyModel":
        model = MyModel(
            learning_rate = learning_rate,
            train_models=["vae", "diffusion", ],
            training = True,
            training_mask = True,
            ddpm_num_steps= 4,
            out_path = str(out_dir),
        )
    else:
        logger.info("please give model_name Patchcore or ReverseDistillation or SimSID")
        return 0

    # model = torch.compile(model)

    print("prepareing Trainer")
    engine = Engine(
        logger = logger,
        callbacks = callbacks,
        default_root_dir= str(out_dir),
        threshold = threshold,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        max_epochs = -1,
    )
    print(f"{engine.image_metric_names=}\n{engine.pixel_metric_names=}")
    # print(f"{engine._cache.args["callbacks"]=}")

    print("starting Training")
    engine.fit(
        datamodule=datamodule,
        model=model,
        ckpt_path = ckpt,
    )


if __name__ == "__main__":

    logging.basicConfig(filename='./logs/debug.log', filemode='w', level=logging.DEBUG)

    # exp_name  = "exp_example"
    exp_name  = "exp_20241208"

    out_dir = "/mnt/c/Users/compbio/Desktop/shimizudata/"
    log_dir  = "./logs"
    in_dir = "/mnt/e/WDDD2_AD"
    dataset_name = "WDDD2_AD"
    model_name = "MyModel"
    target_data = "wildType"
    threshold =  ManualThreshold(default_value=0.5) # "F1AdaptiveThreshold"
    image_metrics  = ['F1Score']# Useless:['MinMax', 'AnomalyScoreDistribution',]
    pixel_metrics = ['AUROC']# Useless:['MinMax', 'AnomalyScoreDistribution',]# 

    version = "v0" # "latest"# "v0" # 
    ckpt = f"{out_dir}/{exp_name}/{model_name}/{dataset_name}/{target_data}/{version}/weights/lightning/model.ckpt"
    ckpt=None

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
        image_metrics  = image_metrics,
        pixel_metrics = pixel_metrics,


        learning_rate  = 1e-1,
        ckpt = ckpt, 
        resolution =  256,
        task = TaskType.SEGMENTATION, #CLASSIFICATION,#
        worker = 30,
        seed  =  44,
        batch = 1,
        debug = True, # False,# 
        debug_data_ratio = 0.08, 
    )

