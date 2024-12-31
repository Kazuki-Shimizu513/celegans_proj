

import torch 
from torchvision.transforms.v2 import Transform
from torchvision.transforms import v2

from pathlib import Path
import json

import wandb
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

from anomalib.data import MVTec
from anomalib import TaskType

from anomalib.engine import Engine
from anomalib.data.utils import TestSplitMode,ValSplitMode

from anomalib.metrics import ManualThreshold

from anomalib.models import (
    Patchcore,
    ReverseDistillation,
)

from celegans_proj.run.image import (
    WDDD2_AD,
    YouTransform,
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
    train_models = ["vae", "diffusion",],
):

    seed_everything(seed, workers=True)

    out_dir = Path(out_dir)
    out_dir = out_dir.joinpath(f"{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir.joinpath("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir.joinpath("images")
    image_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", # pixel_metrics[0], 
        mode='min',
        dirpath = str(model_dir),
        filename='{epoch}',
        save_top_k= 1,
    )
   
    callbacks = [
        checkpoint_callback,
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
                    v2.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225],
                    ),
                    # TODO: YouTransform
                ]) 

    if model_name in ["MyModel", "SimSID"]:
        transforms = v2.Compose([
                        v2.Grayscale(), # Wide resNet has 3 channels
                        v2.PILToTensor(),
                        v2.Resize(
                            size=(resolution, resolution), 
                            interpolation=v2.InterpolationMode.BILINEAR
                        ),
                        v2.ToDtype(torch.float32, scale=True),
                        # TODO: YouTransform
                        # v2.Normalize(mean=[0.485], std=[0.229]),
                    ]) 

    # Initialize the datamodule, model and engine

    # datamodule = MVTec(
    #     root = in_dir ,
    #     category = target_data,
    #     train_batch_size = batch,
    #     eval_batch_size = batch,
    #     num_workers = worker, 
    #     task = task , # TaskType.SEGMENTATION,#CLASSIFICATION,#  
    #     val_split_mode = ValSplitMode.FROM_TEST,
    #     val_split_ratio = 0.1,
    #     test_split_mode = TestSplitMode.FROM_DIR,
    #     test_split_ratio = 0.2,
    #     image_size = (resolution,resolution),
    #     transform = transforms,
    #     seed  = seed,# 44,
    # )
    # datamodule.prepare_data()


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
    print(f"{type(batch['mask'])=}\t{batch['mask'].shape=}\t{len(batch['mask'].unique().tolist())=}")
    print()

    print("prepareing model")

    if model_name == "Patchcore":
        model = Patchcore(
            backbone = "wide_resnet50_2",
            layers = ("layer2", "layer3"),
            pre_trained = True,
            coreset_sampling_ratio = 0.1,
            num_neighbors = 9,
        )
    else:
        msg = f"please give validate model_name, got {model_name=}"
        raise ValueError(msg)

    model = torch.compile(model)

    print("prepareing Trainer")
    engine = Engine(
        logger = logger,
        callbacks = callbacks,
        default_root_dir= str(out_dir),
        threshold = threshold,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        max_epochs = -1,
        log_every_n_steps=1,
    )
    print(f"{engine.image_metric_names=}\n{engine.pixel_metric_names=}")
    print(f"{engine._cache.args["callbacks"]=}")

    print("starting Training")
    engine.fit(
        datamodule=datamodule,
        model=model,
        ckpt_path = ckpt,
    )

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

    image_metrics  = ['AUROC'],
    pixel_metrics = ['AUROC'],
    seed=44,
    debug = False, 
    debug_data_ratio = 0.08, 

    train_models=["vae", "diffusion", ],
):

    seed_everything(seed, workers=True)
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
                        # v2.Normalize(mean=[0.485], std=[0.229]),
                        ]) 


    datamodule = WDDD2_AD(
        root = in_dir,
        category = target_data,
        train_batch_size = batch,
        eval_batch_size = batch,
        num_workers = worker, 
        task =  task, 
        val_split_mode = ValSplitMode.NONE,
        image_size = (resolution,resolution),
        transform = transforms,
        seed  = seed,
        debug = debug, # True, 
        debug_data_ratio = debug_data_ratio, # 0.01, 
        add_anomalous=debug,
    )
    datamodule.setup()

    print("prepareing model")

    for attribute in ("train_data", "val_data", "test_data"):
        data_loader = getattr(datamodule,attribute)
        print(f"total {attribute} iterations: {len(data_loader)}")

    i, batch = next(enumerate(datamodule.test_data))
    print(batch.keys()) # dict_keys(['image_path', 'label', 'image', 'mask'])
    print(f"{type(batch['image_path'])=}\t{batch['image_path']=}")
    print(f"{type(batch['label'])=}\t{batch['label']=}")
    print(f"{type(batch['image'])=}\t{batch['image'].shape=}")
    print(f"{type(batch['mask'])=}\t{batch['mask'].shape=}")
    print()


    if model_name == "Patchcore":
        model = Patchcore(
            backbone = "wide_resnet50_2",
            layers = ("layer2", "layer3"),
            pre_trained = True,
            coreset_sampling_ratio = 0.1,
            num_neighbors = 9,
        )
    else:
        msg = f"please give validate model_name, got {model_name=}"
        raise ValueError(msg)


    model = torch.compile(model)

    print("prepareing Tester")
    engine = Engine(
        logger = logger,
        callbacks = callbacks,
        task = task,
        default_root_dir= str(out_dir),
        threshold = threshold, 
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
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
    logging.basicConfig(filename='./logs/debug_train.log', filemode='w', level=logging.DEBUG)

    exp_name  = "exp_example"

    dataset_name = "WDDD2_AD"
    target_data = "wildType"

#     dataset_name  = "MVTec"
#     target_data = "bottle"

    out_dir = "/mnt/c/Users/compbio/Desktop/shimizudata/"
    in_dir = f"/mnt/e/{dataset_name}"

#     out_dir = "/home/skazuki/result"
#     in_dir = f"/home/skazuki/data/{dataset_name}"

    log_dir  = "./logs"
    model_name = "Patchcore" # "MyModel"
    threshold =   "F1AdaptiveThreshold" # ManualThreshold(default_value=0.5) # 
    image_metrics  = ['F1Score']
    pixel_metrics = ['AUROC']

    ckpt=None

    logger = WandbLogger(
        project =f"{exp_name}",
        name = f"{model_name}_{target_data}",
        save_dir = str(log_dir),
    )


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

        learning_rate  = 1e-8,
        ckpt = ckpt, 
        resolution =  256,
        task = TaskType.SEGMENTATION, #CLASSIFICATION,#
        worker = 16,
        seed  =  44,
        batch = 2, # 2, #12
        debug = False, #
        # debug = True, #
        debug_data_ratio = 0.10, 
    )

    logging.basicConfig(filename='./logs/debug_predict.log', filemode='w', level=logging.DEBUG)

    pseudo_anomaly_modes = [
                            "wildType",
                            "patchBlack",
                            # "gridBlack",
                            "zoom",
                            "shrink",
                            "oneCell",
                        ]
    anomaly_gene_list  = [
                            # "F10E9.8", # sas-4
                            # "F54E7.3", # par-3
                            "C53D5.a", # imb-3
                            "C29E4.8", # let-754
                            "wildType",
                        ]


    exp_name = "exp_example"

    dataset_name = "WDDD2_AD"
    target_data = "wildType"

#     dataset_name  = "MVTec"
#     target_data = "bottle"

    out_dir = Path("/mnt/c/Users/compbio/Desktop/shimizudata/exp_server/")
    in_dir = Path(f"/mnt/e/{dataset_name}")

    log_dir  = Path("./logs")
    # model_names = [ "Patchcore",  "ReverseDistillation",]
    model_names = [ "Patchcore",]
    # model_names = [ "ReverseDistillation",]
    # model_names = ["MyModel"]
    # train_models = ["vae","diffusion",] # 

    threshold =  ManualThreshold(default_value=0.9) # "F1AdaptiveThreshold" #
    image_metrics  = ['F1Score']# Useless:['MinMax', 'AnomalyScoreDistribution',]
    pixel_metrics = ['AUROC']# Useless:['MinMax', 'AnomalyScoreDistribution',]# 

    version = "v1" # "latest"

    for kind in pseudo_anomaly_modes :
    # for kind in anomaly_gene_list :

        for model_name in  model_names :

            logger = WandbLogger(
                project=exp_name,
                name = f"{model_name}_{kind}",
                save_dir = str(log_dir),
            )

            # ckpt_path  = out_dir.joinpath(f"{exp_name}/models/epoch=44.ckpt")
            ckpt_path  = out_dir.joinpath(f"{exp_name}/{model_name}/{dataset_name}/wildType/{version}/weights/lightning/model.ckpt")

            predict(
                exp_name =f"{exp_name}/predict",# /rnai
                out_dir = out_dir,
                in_dir =  in_dir,
                model_name = model_name, 
                target_data = kind, # "wildType",
                threshold =  threshold, 
                logger = logger,
                image_metrics  = image_metrics,
                pixel_metrics = pixel_metrics,


                ckpt = ckpt_path,
                resolution = 256,
                task = TaskType.SEGMENTATION, #CLASSIFICATION,#
                worker = 16,
                seed=44,
                batch = 2,
                debug =  False,# True,#
                debug_data_ratio = 0.1,
            )




