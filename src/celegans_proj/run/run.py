

import argparse 
import logging
from pathlib import Path

import torch 
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from lightning.pytorch.loggers import WandbLogger
import wandb

from anomalib import TaskType

from caenorhabditiselegans_anomalydetection_dataset.run.train import train
from caenorhabditiselegans_anomalydetection_dataset.run.predict import predict 

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")
torch._dynamo.config.suppress_errors = True


def run():
    parser = argparse.ArgumentParser(description='train wildtype and predict for each dataset')

    parser.add_argument('--exp_name', default= "exp_example", help='')
    # parser.add_argument('--exp_name', default= "exp_mvtec", help='')

    # parser.add_argument('--out_dir', default= "/mnt/c/Users/compbio/Desktop/shimizudata/", help='')
    # parser.add_argument('--out_dir', default= "/mnt/e/shimizudata/", help='')
    parser.add_argument('--out_dir', default= "/home/skazuki/playground/CaenorhabditisElegans_AnomalyDetection_Dataset/results/", help='')

    # parser.add_argument('--in_dir', default= "/mnt/e/WDDD2_AD", help='')
    # parser.add_argument('--in_dir', default= "/mnt/e/mvtec_anomaly_detection", help='')
    parser.add_argument('--in_dir', default= "/home/skazuki/data//WDDD2_AD", help='')

    parser.add_argument('--train_ckpt', default= None, help='')
    parser.add_argument('--batch', default= 1, help='')
    # parser.add_argument('--resolution', default= 128,  help='') 
    parser.add_argument('--resolution', default= 256,  help='') 

    parser.add_argument("--pseudo_anomaly_modes", 
                        nargs='+',
                        default=[
                            "wildType",
                            "patchBlack",
                            "gridBlack",
                            "zoom",
                            "shrink",
                            "oneCell",
                        ], type=str)

#     parser.add_argument("--pseudo_anomaly_modes", 
#                         nargs='+',
#                         default=[
#                             "hazelnut",
#                        ], type=str)


    parser.add_argument("--anomaly_gene_list", 
                        nargs='+',
                        default=[
                            "wildType",
                            "F10E9.8", # sas-4
                            "F54E7.3", # par-3
                        ], type=str)
    parser.add_argument("--model_names", 
                        nargs='+',
                        default=[
                            "Patchcore",
                            "ReverseDistillation",
                        ], type=str)



    args = parser.parse_args()

    in_dir = Path(args.in_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir  = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=f"{args.exp_name}",
        name=f"Compare Models",
        config = vars(args),
        dir = str(log_dir),
    )
    # targets = [p.stem for p in in_dir.iterdir() if p.is_dir()] 
    for model_name in tqdm(args.model_names):
        if model_name == "Patchcore":
            resolution = 128
            # resolution = 256 
        else: 
            resolution = args.resolution
        target_data = "wildType"
        # target_data = self.pseudo_anomaly_modes[0]

        logger = WandbLogger(
            project=f"{args.exp_name}",
            name = f"{model_name}_{target_data}",
            save_dir = str(log_dir),
            log_model = "all",#  True, 
        )
        # logger.experiment.config.update(**vars(args))


        print(f"train {model_name}\n")
        train(
            exp_name = args.exp_name,
            out_dir = args.out_dir, 
            in_dir = args.in_dir,
            target_data = target_data,
            model_name = model_name,
            ckpt = args.train_ckpt,
            batch = args.batch,
            resolution = resolution,
            task = TaskType.SEGMENTATION, #CLASSIFICATION,#
            worker = 14,
            logger  = logger, 
            metric_name = "pixel_AUROC",
        )
        for target_data in tqdm(args.pseudo_anomaly_modes):

            logger = WandbLogger(
                project=f"{args.exp_name}",
                name = f"{model_name}_{target_data}",
                save_dir = str(log_dir),
                log_model = "all",#  True, 
            )
            # logger.experiment.config.update(**vars(args))


            print(f"predict {target_data}")
            ckpt_file = out_dir.joinpath(f"{args.exp_name}/{model_name}/WDDD2_AD/wildType/v0/weights/lightning/model.ckpt")
            predict(
                exp_name = f"{args.exp_name}_1",
                out_dir = args.out_dir, 
                in_dir = args.in_dir,
                target_data = target_data,
                model_name = model_name,
                ckpt = ckpt_file,
                batch = args.batch,
                resolution = args.resolution,
                task = TaskType.SEGMENTATION, #CLASSIFICATION,#
                worker = 14,
                logger  = logger, 
            )

#             for target_data in tqdm(args.anomaly_gene_list):

            logger = WandbLogger(
                project=f"{args.exp_name}",
                name = f"{model_name}_{target_data}",
                save_dir = str(log_dir),
                log_model = "all",#  True, 
            )
            # logger.experiment.config.update(**vars(args))

#                 print(f"predict {target_data}")
#                 ckpt_file = out_dir.joinpath(f"results/{model_name}/WDDD2_AD/WildType/v0/weights/lightning/model.ckpt")
#                 predict(
#                     exp_name = args.exp_name,
#                     out_dir = args.out_dir, 
#                     in_dir = args.in_dir,
#                     target_data = target_data,
#                     model_name = model_name,
#                     ckpt = ckpt_file,
#                     batch = args.batch,
#                     resolution = args.resolution,
#                     task = TaskType.CLASSIFICATION,#SEGMENTATION, #
#                     worker = 30,
#                     logger  = logger, 
#                 )
     

    print("finished")
if __name__ == "__main__":

    logging.basicConfig(filename='./logs/debug.log', filemode='w', level=logging.DEBUG)
    run()

