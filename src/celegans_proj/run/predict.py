
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

    anomaly_data_list = [
                            "wildType",
                            "patchBlack",
                            "gridBlack",
                            "zoom",
                            "shrink",
                            "oneCell",
                        ],

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

    max_epochs = -1 ,
    learning_rate  = 1e-8,
    train_models=["vae", "diffusion", ],
):

    seed_everything(seed, workers=True)
    out_dir = Path(out_dir)
    out_dir = out_dir.joinpath(f"{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir.joinpath("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_name in ["MyModel", "SimSID"]:
        # checkpoint_callback = ModelCheckpoint(
        #     monitor="val_loss", # pixel_metrics[0], 
        #     mode='min',
        #     dirpath = str(model_dir),
        #     filename='{epoch}',
        #     save_top_k= 1,
        # )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_score", # pixel_metrics[0], 
            mode='max',
            dirpath = str(model_dir),
            filename='{epoch}',
            save_top_k= 1,
        )
        # checkpoint_callback = ModelCheckpoint(
        #     monitor="pixel_AUROC", # pixel_metrics[0], 
        #     mode='max',
        #     dirpath = str(model_dir),
        #     filename='{epoch}',
        #     save_top_k= 1,
        # )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="pixel_AUROC", # pixel_metrics[0], 
            mode='max',
            dirpath = str(model_dir),
            filename='{epoch}',
            save_top_k= 1,
        )

    callbacks = [
        checkpoint_callback,
        EarlyStopping("val_score"),
        # DeviceStatsMonitor(),
    ]

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
            train_models=train_models,
            training = False, # True, #
            training_mask = True, 
            ddpm_num_steps= 10, # 100, # 50, # 
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
        max_epochs = max_epochs, # -1,
        log_every_n_steps=1,
    )
 
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

    for target_data in anomaly_data_list:

        print(f"{target_data=}")

        if target_data == "wildType":
            debug_data_ratio *= 0.1 

        datamodule = WDDD2_AD(
            root = in_dir,
            category = target_data,
            train_batch_size = 2,
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


        if target_data == "wildType":
            print("starting Training")
            engine.fit(
                # datamodule=datamodule,
                # train_dataloaders=datamodule.train_dataloader(),
                train_dataloaders=datamodule.test_dataloader(),
                val_dataloaders=datamodule.val_dataloader(),
                model=model,
                ckpt_path = ckpt,
            )
            ckpt_path  = out_dir.joinpath(f"{exp_name}/{model_name}/{dataset_name}/wildType/latest/weights/lightning/model.ckpt")

        print("starting predictions")
        predictions = engine.predict(
            model = model,
            datamodule = datamodule,
            # dataloaders=datamodule.val_dataloader(),
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

    # pseudo_anomaly_modes = [
    #                         "shrink",
    #                         "shrink60",
    #                         "shrink70",
    #                         "shrink80",
    #                         "shrink90",
    #                         "wildType",
    #                         "zoom110",
    #                         "zoom120",
    #                         "zoom130",
    #                         "zoom140",
    #                         "zoom",
    #                     ]
 
    anomaly_gene_list  = [
                            "wildType",
                            # "F10E9.8", # sas-4
                            # "F54E7.3", # par-3
                            "C53D5.a", # imb-3
                            "C29E4.8", # let-754
                        ]


    exp_name = "exp_20250118"
    # exp_name = "exp_20250113_ReverseDistillation"
    # exp_name = "exp_20250105_diffusion"
    # exp_name = "exp_20241229_vae"

    dataset_name = "WDDD2_AD"
    target_data = "wildType"

#     dataset_name  = "MVTec"
#     target_data = "bottle"

    # out_dir = Path("/mnt/c/Users/compbio/Desktop/shimizudata/exp_server/")
    out_dir = Path("/mnt/c/Users/compbio/Desktop/shimizudata/")
    in_dir = Path(f"/mnt/e/{dataset_name}")

    # out_dir = Path("/home/skazuki/result")
    # in_dir = Path(f"/home/skazuki/data/{dataset_name}")

    log_dir  = Path("./logs")

    # model_names = [ "Patchcore",  "ReverseDistillation",]
    # model_names = [ "Patchcore",]
    # model_names = [ "ReverseDistillation",]
    model_names = ["MyModel"]
    train_models = ["vae",] # "diffusion",  #

    threshold = "F1AdaptiveThreshold" # ManualThreshold(default_value=0.9) # 
    image_metrics  = ['F1Score']# Useless:['MinMax', 'AnomalyScoreDistribution',]
    pixel_metrics = ['AUROC']# Useless:['MinMax', 'AnomalyScoreDistribution',]# 

    version = "latest" # "v5" # 

    mode = "pseudo"
    # mode = "RNAi"

    logger = WandbLogger(
        project=f"{exp_name}",
        name = f"{model_names[0]}_{target_data}",
        save_dir = str(log_dir),
    )


    # for kind in pseudo_anomaly_modes :
    # for kind in anomaly_gene_list :

    for model_name in  model_names :

        ckpt_path  = out_dir.joinpath(f"{exp_name}/models/epoch=1705.ckpt") # diffusion
        # ckpt_path  = out_dir.joinpath(f"{exp_name}/models/epoch=891.ckpt") # server vae
        # ckpt_path  = out_dir.joinpath(f"{exp_name}/models/epoch=1172.ckpt") # server diffusion
        # ckpt_path  = out_dir.joinpath(f"{exp_name}/{model_name}/{dataset_name}/wildType/{version}/weights/lightning/model.ckpt")

        predict(
            exp_name =f"{exp_name}/{mode}",
            out_dir = out_dir,
            in_dir =  in_dir,
            model_name = model_name, 
            target_data = "wildType",
            threshold =  threshold, 
            logger = logger,
            image_metrics  = image_metrics,
            pixel_metrics = pixel_metrics,

            ckpt = ckpt_path,
            resolution = 256,
            task = TaskType.SEGMENTATION, #CLASSIFICATION,#
            worker = 16,
            seed = 44,
            batch = 2, # 70, # 500, # 200,
            debug =  True, # False, # 
            debug_data_ratio = 0.999,
            train_models = train_models,
            learning_rate  = 1e-1000,

            anomaly_data_list = pseudo_anomaly_modes,
        )

