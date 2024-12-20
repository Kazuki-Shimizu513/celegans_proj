
import torch 
from torchvision.transforms.v2 import Transform
from torchvision.transforms import v2

from pathlib import Path

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

from celegans_proj.run.image.wddd2 import (
    WDDD2_AD,
    YouTransform,
) 
from celegans_proj.run.train import train

from celegans_proj.models import (
    SimSID,
    MyModel,
)

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

    image_metrics  = ['AUROC'],
    pixel_metrics = ['AUROC'],
    seed=44,
    debug = False, 
    debug_data_ratio = 0.08, 

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
                        v2.Normalize(mean=[0.485], std=[0.229]),
                        ]) 


    datamodule = WDDD2_AD(
        root = in_dir,
        category = target_data,
        train_batch_size = batch,
        eval_batch_size = batch,
        num_workers = worker, 
        task =  task, 
        val_split_mode = ValSplitMode.SAME_AS_TEST,
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
            learning_rate = 1e-8,
            train_models=["vae", "diffusion", ],
            training = True,
            training_mask = True,
            out_path = str(out_dir),
        )
 
    else:

        logger.info("please give model_name Patchcore or ReverseDistillation")

    # model = torch.compile(model)

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

    logging.basicConfig(filename='./logs/debug.log', filemode='w', level=logging.DEBUG)

    pseudo_anomaly_modes = [
                            "wildType",
                            "patchBlack",
                            "gridBlack",
                            "zoom",
                            "shrink",
                            "oneCell",
                        ]
    anomaly_gene_list  = [
                            "wildType",
                            # "F10E9.8", # sas-4
                            # "F54E7.3", # par-3
                            "C53D5.a", # imb-3
                            "C29E4.8", # let-754
                        ]


    # exp_name = "exp_example"
    exp_name  = "exp_20241213"

    dataset_name = "WDDD2_AD"
    target_data = "wildType"

#     dataset_name  = "MVTec"
#     target_data = "bottle"

    # out_dir = Path("/mnt/c/Users/compbio/Desktop/shimizudata/exp_server/")
    # in_dir = Path(f"/mnt/e/{dataset_name}")

    out_dir = Path("/home/skazuki/result")
    in_dir = Path(f"/home/skazuki/data/{dataset_name}")

    log_dir  = Path("./logs")
    # model_names = [ "Patchcore",  "ReverseDistillation",]
    # model_names = [ "Patchcore",]
    # model_names = [ "ReverseDistillation",]
    model_names = ["MyModel"]

    threshold =  ManualThreshold(default_value=0.9) # "F1AdaptiveThreshold" #
    image_metrics  = ['F1Score']# Useless:['MinMax', 'AnomalyScoreDistribution',]
    pixel_metrics = ['AUROC']# Useless:['MinMax', 'AnomalyScoreDistribution',]# 

    version = "v1" # "latest"

    # for kind in pseudo_anomaly_modes :
    for kind in anomaly_gene_list :

        for model_name in  model_names :

            logger = WandbLogger(
                project=exp_name,
                name = f"{model_name}_{kind}",
                save_dir = str(log_dir),
            )

            ckpt_path  = out_dir.joinpath(f"{exp_name}/models/epoch=50.ckpt")
            # ckpt_path  = out_dir.joinpath(f"{exp_name}/{model_name}/{dataset_name}/wildType/{version}/weights/lightning/model.ckpt")

            predict(
                exp_name =f"{exp_name}/predict_rnai",
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
                batch = 10,
                debug =  False,# True,#
                debug_data_ratio = 0.1,
            )

