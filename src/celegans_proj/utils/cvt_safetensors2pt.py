
from pathlib import Path
import torch 
from torchvision.transforms import v2
import safetensors
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch import Trainer, seed_everything
from anomalib import TaskType
from anomalib.data.utils import ValSplitMode



from celegans_proj.models import (
    MyModel,
)
from celegans_proj.run.image import (
    WDDD2_AD,
    YouTransform,
)

seed_everything(42) 
torch.set_float32_matmul_precision("medium")
torch._dynamo.config.suppress_errors = True

if __name__ == "__main__":
    exp_name = "exp_202523"
    base_path = Path(f"/mnt/c/Users/compbio/Desktop/shimizudata/{exp_name}")
    ckpt_safetensors_name = "checkpoint-992495-best-LDM-v2-MIRU2024-M2中間"
    ckpt_pt_name = "checkpoint-992495-pt"
    safetensors_dir = base_path.joinpath(f"{ckpt_safetensors_name}")
    pt_dir = base_path.joinpath(f"{ckpt_pt_name}")
    pt_dir.mkdir(parents=True, exist_ok=True)

    print(str(safetensors_dir))
    print(str(pt_dir))

#     for safetensors_path in safetensors_dir.iterdir():
#         f_name = safetensors_path.stem
#         if not f_name.startswith("model"):
#             continue
#         f_path = pt_dir.joinpath(f"{f_name}.pt")


#         loaded = safetensors.torch.load_file(safetensors_path)
#         torch.save(loaded, f_path)

#         print(f"{f_name}\t{loaded.keys()=}\n\n")

    my_model = MyModel(
            learning_rate = 1e-8,
            train_models= ["vae",],#  "diffusion"
            training = True,
            training_mask = True,
            ddpm_num_steps= 10, 
            out_path = str(base_path),
    )

    for safetensors_path in safetensors_dir.iterdir():
        f_name = safetensors_path.stem
        if not f_name.startswith("model-"):
            continue
        f_path = pt_dir.joinpath(f"{f_name}.pt")
        checkpoint = torch.load(f_path)

        if f_name.startswith("model-1"):
            my_model.model.pipe.vae.load_state_dict(ckeckpoint)
        if f_name.startswith("model-2"):
            my_model.model.pipe.unet.load_state_dict(ckeckpoint)

    model = torch.compile(my_model)

    # Train in order to save checkpoints
    logger = WandbLogger(
        project=f"{exp_name}",
        name = "LDM",
        save_dir = "./logs",
    )



    transforms = v2.Compose([
                    v2.Grayscale(), # Wide resNet has 3 channels
                    v2.PILToTensor(),
                    v2.Resize(
                        size=(256, 256), 
                        interpolation=v2.InterpolationMode.BILINEAR
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    # TODO: YouTransform
                    # v2.Normalize(mean=[0.485], std=[0.229]),
                ]) 

    datamodule = WDDD2_AD(
        root = "/mnt/e/WDDD2_AD/",
        category = "wildType",
        train_batch_size = 1,
        eval_batch_size = 1,
        num_workers = 16, 
        task = TaskType.SEGMENTATION,#CLASSIFICATION,#  
        val_split_mode = ValSplitMode.SYNTHETIC, #.FROM_TEST, # 
        val_split_ratio = 0.1,
        image_size = (256,256),
        transform = transforms,
        seed  = 44,
        debug = True, 
        debug_data_ratio =  0.1, 
        add_anomalous=False,
    )
 
    datamodule.setup()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_score", # pixel_metrics[0], 
        mode='max',
        dirpath = str(pt_dir),
        filename='{epoch}',
        save_top_k= 1,
    )

    trainer = Trainer(
        logger=logger,
        default_root_dir=str(base_path),
        callbacks=[checkpoint_callback,],
        max_epochs=1,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        reload_dataloaders_every_n_epochs=1,
    )

    trainer.fit(
        datamodule=datamodule,
        model=my_model, 
   )

