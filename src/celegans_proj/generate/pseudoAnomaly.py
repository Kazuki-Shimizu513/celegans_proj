


import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import itertools

# from abc import ABC
import os
from typing import Union
from collections.abc import Sequence

from pathlib import Path
from shutil import copy

# from pandas import DataFrame
import pandas as pd 
# from tabulate import tabulate

import numpy as np
# import cv2 as cv
from PIL import Image, ImageStat
import albumentations as A

# from sklearn.model_selection import train_test_split

# import tifffile
# import h5py

from celegans_proj.utils.table_builder.celegans import (
    WildTypeCelegansTableBuilder, 
    RNAiCelegansTableBuilder,
)
from celegans_proj.utils.file_import import WDDD2FileNameUtil
from celegans_proj.generate.transfer import Transporter

logger = logging.getLogger(__name__)


'''
    [alumentations](https://albumentations.readthedocs.io/en/latest/)
    [alumentations_examples](https://github.com/albumentations-team/albumentations_examples)
    [albumentations github](https://github.com/albumentations-team/albumentations#benchmarking-results)
'''


class PseudoAnomalyGenerator:
    def __init__(
        self,
        datatable_path: Union[ str | os.PathLike ], # wildtype_test.csv
        out_path: Union[ str | os.PathLike ] = Path('~/WDDD2_AD'),
        in_dir:Union[ str | os.PathLike ] = Path('~/WDDD2/TIF_GRAY')
    ) -> None:
        self.out_path = Path(out_path)
        self.in_dir = Path(in_dir)

        df = pd.read_csv(datatable_path)
        self.img_path_list = df["img_path"].to_list()
        self.BD5_path_list = df["h5_path"].unique()
        Z_list = df["Z"].astype('int').unique()
        self.Z_range = tuple([min(Z_list), max(Z_list)])

    def __call__(
        self,
        pseudo_anomaly_kinds = [
            "patchBlack",
            "gridBlack",
            "shrink",
            "zoom",
            "oneCell",
            ],
    ) -> None:
        self.generate(
            pseudo_anomaly_kinds,
        )

    def generate(
        self,
        pseudo_anomaly_kinds: Sequence[str] = [
            "patchBlack",
            "gridBlack",
            "shrink",
            "zoom",
            "oneCell",
        ],
    ) -> None:

        '''

          WDDD2_AD
          ├── zoom
          │   ├── (defects_config.json)
          │   ├── ground_truth
          │   │   └── anomaly 
          │   │       └── pseudoAnomaly_zoom_081007_02_T53_Z36_mask.tiff
          │   ├── train # (empty)
          │   ├── test 
          │   │   └── anomaly
          │   │       └── pseudoAnomaly_zoom_081007_02_T53_Z36.tiff
          │   └── validation # (empty)
          ├──  ...

        '''

        out_path = Path(self.out_path)

        Transporter.build_WDDD2_AD_dir_structure(
            out_path, 
            kinds=pseudo_anomaly_kinds
        )

        if "oneCell" in pseudo_anomaly_kinds:
            _target_dir = out_path.joinpath("oneCell")
            target_dir = _target_dir.joinpath("test/anomaly")
            target_dir_2 = _target_dir.joinpath("ground_truth/anomaly")

            logger.info("start fetch_other_cellStage_images")
            img_paths = self.fetch_other_cellStage_images(
                target_cellStage=1
            )

            logger.info("saving other_cellStage_images")
            with logging_redirect_tqdm(loggers=[logger]):
                for img_path in tqdm(img_paths):

                    [base_dir, filename, suffix] = WDDD2FileNameUtil.strip_path(img_path)
                    [kotai_kind, kotai_name, T, Z,] = WDDD2FileNameUtil.strip_name(filename)
                    e = kotai_name.split("_")
                    e[0] = "pseudoAnomaly"
                    e[1] = "oneCell"
                    e.append(f"T{str(T)}")
                    e.append(f"Z{str(Z)}")
                    name = "_".join(e)
                    img = Image.open(img_path)
                    self.save_img(img, target_dir, name,)

                    msk = Image.fromarray(np.zeros_like(np.array(img)))
                    e.append("mask")
                    name = "_".join(e)
                    self.save_img(msk, target_dir_2, name,)


        logger.info(f"start generating pseudoAnomaly")
        with logging_redirect_tqdm(loggers=[logger]):
            for img_path in tqdm(self.img_path_list):
                img_path = Path(img_path)

                [base_dir, filename, suffix] = WDDD2FileNameUtil.strip_path(img_path)
                [kotai_kind, kotai_name, T, Z,] = WDDD2FileNameUtil.strip_name(filename)
                e = kotai_name.split("_")
                e[0] = "pseudoAnomaly"
                e.append(f"T{str(T)}")
                e.append(f"Z{str(Z)}")

                for kind in pseudo_anomaly_kinds:
                    if kind == "oneCell":
                        continue
                    img , msk = self.generate_kind(img_path, kind)
                    self.save_img_and_msk(img, msk, e, kind, out_path)

    def generate_kind(
        self,
        img_path,
        kind,
    ) -> None:
        if kind == "patchBlack":
            img, msk = self.generate_patchBlack(img_path)
        if kind == "gridBlack":
            img, msk = self.generate_gridBlack(img_path)
        if  "zoom" in kind:
            if len(kind) != 4:
                rate = float(kind[-3:]) * 0.01
            else:
                rate = 1.5

            img, msk = self.generate_zoom(img_path, rate=rate)
        if "shrink" in kind :
            if len(kind) != 6:
                rate = float(kind[-2:]) * 0.01
            else:
                rate = 0.5

            img, msk = self.generate_shrink(img_path, rate=rate)
        return img ,msk

    def generate_patchBlack(
        self,
        img_path:Union[ str | os.PathLike ],
        patch_num :Sequence[int]= (100, 100),
        patch_h :Sequence[int]= (10, 10),
        patch_w :Sequence[int]= (10, 10),
        fill_value: int = 0,
        resolution: int = 600,
    ) -> Sequence[Image]:
        transform = A.Compose(
            [
                A.CoarseDropout(
                    min_holes=patch_num[0],max_holes=patch_num[1],
                    min_height=patch_h[0],max_height=patch_h[1],
                    min_width=patch_w[0],max_width=patch_w[1],
                    fill_value=fill_value,
                    p=1,
                ),
                A.Resize(resolution, resolution,
                    p=1,
                ),
            ]
        )

        img = Image.open(str(img_path))

        image = np.array(img).astype(np.uint8)
        mask = np.full_like(image,0,dtype=np.uint8)

        augmented = transform(image=image, mask=mask)
        img = Image.fromarray(augmented["image"])

        diff = image - augmented["image"]
        msk = np.where(diff > 0, 255, mask)
        msk = Image.fromarray(msk)

        # print(img.getextrema(), msk.getextrema())

        return img,msk

    def generate_gridBlack(
        self,
        img_path:Union[ str | os.PathLike ],
        grid_num :Sequence[int]= (10, 10),
        grid_h :Sequence[int]= (30, 30),
        grid_w :Sequence[int]= (30, 30),
        fill_value: int = 0,
        resolution: int = 600,
    ) -> Sequence[Image]:
        transform = A.Compose(
            [
                A.CoarseDropout(
                    min_holes=grid_num[0],max_holes=grid_num[1],
                    min_height=grid_h[0],max_height=grid_h[1],
                    min_width=grid_w[0],max_width=grid_w[1],
                    fill_value=fill_value,
                    p=1,
                ),
                A.Resize(resolution, resolution),
            ]
        )

        img = Image.open(str(img_path))

        image = np.array(img).astype(np.uint8)
        mask = np.full_like(image,0,dtype=np.uint8)

        augmented = transform(image=image, mask=mask)
        img = Image.fromarray(augmented["image"])

        diff = image - augmented["image"]
        msk = np.where(diff > 0, 255, mask)
        msk = Image.fromarray(msk)


        return img,msk

    def generate_shrink(
        self,
        img_path:Union[ str | os.PathLike ],
        rate: float = 0.5,
    ) -> Sequence[Image]:
        image = Image.open(str(img_path))

        # A.shiftscalerotate でもいけるっぽい
        h, w = image.size
        shrink = image.resize(
            (int(w * rate), int(h * rate)), 
            resample=Image.BILINEAR,  #  Image.NEAREST,
        )
        shrink = self.add_margin(shrink, w, h)
        img = self.crop_center(shrink, w, h)
        img = np.array(img).astype(np.uint8)

        msk = np.full_like(img,0,dtype=np.uint8)
        diff = image - img 
        msk = np.where(diff > 0, 255, msk)
        msk = Image.fromarray(msk)

        img = Image.fromarray(img)


        return img,msk

    def generate_zoom(
        self,
        img_path:Union[ str | os.PathLike ],
        rate: float = 1.5,
    ) -> Sequence[Image]:
        image = Image.open(str(img_path))

        # A.shiftscalerotate でもいけるっぽい
        h, w = image.size
        zoom = image.resize(
            (int(w * rate), int(h * rate)),
            resample=Image.BILINEAR,  #  Image.NEAREST,
        )
        img = self.crop_center(zoom, w, h)
        img = np.array(img).astype(np.uint8)


        msk = np.full_like(img,0,dtype=np.uint8)
        diff = image - img 
        msk = np.where(diff > 0, 255, msk)
        msk = Image.fromarray(msk)

        img = Image.fromarray(img)



        return img,msk


    @staticmethod
    def crop_center(
        pil_img, 
        crop_width, 
        crop_height
    ) -> Image:
        img_width, img_height = pil_img.size
        return pil_img.crop(
            (
                (img_width - crop_width) // 2,
                (img_height - crop_height) // 2,
                (img_width + crop_width) // 2,
                (img_height + crop_height) // 2,
            )
        )

    @staticmethod
    def add_margin(
        pil_img, 
        new_width, 
        new_height
    ):
        width, height = pil_img.size
        stat = ImageStat.Stat(pil_img)
        color = int(stat.median[0])
        color = pil_img.getpixel((20, 20))
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result





    def fetch_other_cellStage_images(
        self,
        target_cellStage: int = 1,
    ) -> Sequence[ str | os.PathLike ]:
        BD5_path_list = self.BD5_path_list
        in_dir = self.in_dir

        cellStage_img_paths = []
        with logging_redirect_tqdm(loggers=[logger]):
            for BD5_path in tqdm(BD5_path_list):
                BD5_path = Path(BD5_path)
                kotai_name = "_".join(BD5_path.stem.split("_")[:-1]) # _BD5 をとる
                _in_dir = in_dir.joinpath(kotai_name)

                cell_stage_img_paths = WDDD2FileNameUtil.fetch_Ts_from_cellStage(
                    BD5_path, # 個体ごと
                    in_dir = _in_dir,
                    target_cellStage = target_cellStage, 
                    Z_range = self.Z_range, # (35,40),
                )
                cellStage_img_paths.extend(cell_stage_img_paths)


        return cellStage_img_paths



    def save_img_and_msk(
        self,
        img: Image,
        msk: Image,
        e: Sequence[str],
        kind: str,
        out_path: Union[ str | os.PathLike ],
    ) -> None:

        _target_dir = out_path.joinpath(kind)
        target_dir = _target_dir.joinpath("test/anomaly")
        target_dir.mkdir(parents=True, exist_ok=True)

        e[1] = kind 
        name = "_".join(e)
        self.save_img(img, target_dir, name,)

        target_dir = _target_dir.joinpath("ground_truth/anomaly")
        target_dir.mkdir(parents=True, exist_ok=True)

        _e = e.copy()
        _e.append("mask")
        name = "_".join(_e)
        self.save_img(msk, target_dir, name,)


    def save_img(
        self,
        img: Image,
        target_dir: Union[ str | os.PathLike ],
        name: str,
        suffix: str = ".tiff",
    ) -> None:
        out_dir = Path(target_dir)
        out_path = out_dir.joinpath(name+suffix)
        img.save(out_path)


def test_generate_patchBlack():

    out_path = Path('/mnt/c/Users/compbio/Desktop/shimizudata/test')
    out_path.mkdir(parents=True, exist_ok=True)

    # D:\WDDD2\TIF_GRAY\wt_N2_081007_02\wt_N2_081007_02_T53_Z37.tiff
    img_path =  img_data_path.joinpath("wt_N2_081007_02/wt_N2_081007_02_T53_Z37.tiff")

    [base_dir, filename, suffix] = WDDD2FileNameUtil.strip_path(img_path)
    [kotai_kind, kotai_name, T, Z,] = WDDD2FileNameUtil.strip_name(filename)
    e = kotai_name.split("_")
    e[0] = "pseudoAnomaly"
    e.append(f"T{str(T)}")
    e.append(f"Z{str(Z)}")

    img, msk = generator.generate_patchBlack(
                    img_path,
                    patch_num = (100, 100),
                    patch_h = (10, 10),
                    patch_w = (10, 10),
                    fill_value = 0,
                    resolution = 600,
                )

    generator.save_img_and_msk(img, msk, e, 'patchBlack', out_path)

    out_path1 = out_path.joinpath("img.tiff")
    img.save(out_path1)
    out_path2 = out_path.joinpath("mask.tiff")
    msk.save(out_path2)




if __name__ == "__main__":

    logging.basicConfig(filename='./logs/debug.log', filemode='w', level=logging.DEBUG)

    in_data_path = Path('/mnt/d/WDDD2')
    img_data_path = in_data_path.joinpath('TIF_GRAY')
    h5_data_path = in_data_path.joinpath('BD5')
    out_data_path = Path('/mnt/e/WDDD2_AD')


    datatable_path = out_data_path.joinpath('meta_data/datatable/wildType_test.csv')
    pseudo_anomaly_kinds = [
        "patchBlack",
        "gridBlack",
        "shrink",
        "zoom",
        "oneCell",
    ]
    pseudo_anomaly_kinds = [
        "shrink", # 50%
        "shrink60",
        "shrink70",
        "shrink80",
        "shrink90",
        "zoom110",
        "zoom120",
        "zoom130",
        "zoom140",
        "zoom", # 150%
    ]

    generator = PseudoAnomalyGenerator(
        datatable_path, 
        out_path= out_data_path,
        in_dir= img_data_path 
    )

    generator(pseudo_anomaly_kinds)

    # test_generate_patchBlack()






