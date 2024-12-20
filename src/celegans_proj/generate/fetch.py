


import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import itertools

from readlif.reader import LifFile
import requests

import os
# from abc import ABC
from typing import Union
from collections.abc import Sequence

from pathlib import Path
from shutil import copy

# from pandas import DataFrame
import pandas as pd 
from tabulate import tabulate
import numpy as np

import cv2 as cv
from PIL import Image
import albumentations as A


import tifffile
import h5py

from celegans_proj.utils.table_builder.celegans import (
    WildTypeCelegansTableBuilder,
    RNAiCelegansTableBuilder,
)
from celegans_proj.utils.file_import import (
    WDDD2FileNameUtil,
)
logger = logging.getLogger(__name__)


class WDDD2fetcher:
    def __init__(
        self,
        wddd2_url="https://wddd.riken.jp/",
        out_dir="/home/skazuki/data/WDDD2/"
    ) -> None:
        self.out_dir = Path(out_dir)

    def __call__(
        self,
        address,
        bdml,
    ):
        print("download lif files")
        for url in tqdm(address):
            out = self.split_wddd_url(url) 
            kind_dir = self.out_dir.joinpath("LIF")
            orf_dir = kind_dir.joinpath(out["ORF"])
            orf_dir.mkdir(mode=0o777,parents=True, exist_ok=True)
            filename = orf_dir.joinpath(out["name"]+".lif")
            # if filename.is_file():
            #     continue
            self.download(str(url), str(filename), chunk_size=1024)

        print("download bd5 bdml h5 files")
        for url in tqdm(bdml):
            out = self.split_wddd_url(url) 
            h5_dir = self.out_dir.joinpath("BD5")
            h5_dir.mkdir(mode=0o777,parents=True, exist_ok=True)
            filename = h5_dir.joinpath(out["name"]+".h5")
            # if filename.is_file():
            #     continue
            self.download(str(url), str(filename), chunk_size=1024)

    @staticmethod
    def download(url: str, fname: str, chunk_size=1024):
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

    # TODO: move to WDDD2FileNameUtil
    @staticmethod
    def split_wddd_url(url):
        lif_name = url.split("/")[-1]
        name = list(lif_name.split(".")[:-1])
        if len(name)>1:
            name = ".".join(name)
        e = name.split("_")
        return dict(name=name, kind=e[0], ORF=e[1], day=e[2], num=e[3])

class WDDD2expander:
    def __init__(
        self,
        wddd2_url="https://wddd.riken.jp/",
        out_dir="/home/skazuki/data/WDDD2/"
    ) -> None:
        self.out_dir = Path(out_dir)

    def __call__(
        self,
        address,
        bdml,
    ):
        for url in tqdm(address):
            out = WDDD2fetcher.split_wddd_url(url) 
            kind_dir = self.out_dir.joinpath("LIF")
            orf_dir = kind_dir.joinpath(out["ORF"])
            filename = orf_dir.joinpath(out["name"]+".lif")
            file = LifFile(str(filename))
            # print(file) # 1 image
            lif_image = file.get_image(img_n=0)
            # print(lif_image.info)
            vol = self.wddd2_lif_to_numpy_stack(lif_image)
            # print(vol.shape)# (360, 66, 600, 600) ==(T,Z,H,W) 

            # TODO: check upside-down
            # TODO: visualize voxel and save using ../visualize/visualizd_voxel.py 

            # save volume
            npy_dir = self.out_dir.joinpath("NPY")
            npy_dir.mkdir(mode=0o777,parents=True, exist_ok=True)
            fname = npy_dir.joinpath(out["name"]+".npy")
            if not fname.is_file():
                np.save(str(fname), vol)

            # save as img foreach timepoint and depth 
            tif_dir = self.out_dir.joinpath("TIFF")
            name_dir = tif_dir.joinpath(out["name"])
            name_dir.mkdir(mode=0o777,parents=True, exist_ok=True)
            T,Z,W,H = vol.shape
            for (t, z) in tqdm(tuple(itertools.product(range(T),range(Z)))):
                fname = name_dir.joinpath(out["name"]+f"_T{t}_Z{z}.tiff")
                if fname.is_file():
                    continue
                img = Image.fromarray(np.squeeze(vol[t,z,:,:]).copy())
                img.save(str(fname), quality=95)


    @staticmethod
    def wddd2_lif_to_numpy_stack(lif_image):
        num_slices = lif_image.info['dims'].z
        num_timepoints = lif_image.info['dims'].t
        # make ndarray shape (T,Z,W,H)
        return np.asarray([[
            np.array(lif_image.get_frame(z=z, t=t)) \
                for z in range(num_slices)
            ] for t in range(num_timepoints)
        ])


if __name__ == "__main__":
    print(f"in fetch.py")

    out_dir= "/home/skazuki/data/WDDD2_tmp/" # Parmission Denided : WDDD2 is owned by Mr. 1009 

    imb_3 = [
        "https://wddd.riken.jp/LIF/RNAi_C53D5.a_120606_01.lif",
        "https://wddd.riken.jp/LIF/RNAi_C53D5.a_120606_02.lif",
    ]
    imb_3_h5 = [
        "https://wddd.riken.jp/BD5/RNAi_C53D5.a_120606_01_bd5.h5",
        "https://wddd.riken.jp/BD5/RNAi_C53D5.a_120606_02_bd5.h5",
    ]
    let_754 = [
        "https://wddd.riken.jp/LIF/RNAi_C29E4.8_100526_01.lif",
        "https://wddd.riken.jp/LIF/RNAi_C29E4.8_100526_02.lif",
        "https://wddd.riken.jp/LIF/RNAi_C29E4.8_100526_03.lif",
        "https://wddd.riken.jp/LIF/RNAi_C29E4.8_100526_04.lif",
        "https://wddd.riken.jp/LIF/RNAi_C29E4.8_100526_05.lif",
        "https://wddd.riken.jp/LIF/RNAi_C29E4.8_101008_01.lif",
        "https://wddd.riken.jp/LIF/RNAi_C29E4.8_101008_02.lif",
        "https://wddd.riken.jp/LIF/RNAi_C29E4.8_101008_03.lif",
        "https://wddd.riken.jp/LIF/RNAi_C29E4.8_101008_04.lif",
    ]
    let_754_h5 = [
    ]

    RNAi_address = [*imb_3,*let_754,]
    RNAi_bdml = [*imb_3_h5,*let_754_h5,]

    fetcher = WDDD2fetcher(
                    wddd2_url="https://wddd.riken.jp/",
                    out_dir=out_dir,
                )
    expander = WDDD2expander(
                    wddd2_url="https://wddd.riken.jp/",
                    out_dir=out_dir,
                )

    fetcher(RNAi_address, RNAi_bdml,)
    expander(RNAi_address, RNAi_bdml, )

#     fetcher([RNAi_address[0]],[RNAi_bdml[0]],)
#     expander([RNAi_address[0]], [RNAi_bdml[0]], )




