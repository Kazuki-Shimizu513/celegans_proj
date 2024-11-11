
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import os
from typing import Union
from collections.abc import Sequence

from pathlib import Path

import numpy as np
import pandas as pd 

# from pandas import DataFrame
# from tabulate import tabulate

import h5py

logger = logging.getLogger(__name__)

class WDDD2FileNameUtil:
    def __init__(
        self,
    ) -> None:
        pass

    # '/mnt/d/WDDD2/TIF_GRAY/wt_N2_081113_01/wt_N2_081113_01_T73_Z35.tiff',
    # D:\WDDD2\TIF_GRAY\RNAi_F54E7.3_100706_05\RNAi_F54E7.3_100706_05_T11_Z55.tiff
    @staticmethod
    def strip_path(
        path: Union[str, os.PathLike],
    ) -> Sequence[str]:
        p = Path(path)
        suffix = p.suffix 
        filename = p.stem 
        base_dir = str(p.parent)
        return [base_dir, filename, suffix]

    @staticmethod
    def strip_name(
        filename: str, 
    ) -> Sequence[str]:
        e = filename.split("_")
        Z = str(e[-1][1:])
        T = str(e[-2][1:])
        kotai_kind = WDDD2FileNameUtil.get_kotai_kind(filename)
        kotai_name = "_".join(e[:-2])
        return [kotai_kind, kotai_name, T, Z,]

    @staticmethod
    def get_kotai_kind(kotai_name):
        kotai_name_el = kotai_name.split("_")
        if kotai_name_el[0]== "pseudoAnomaly":
            kind = kotai_name_el[1]
        elif kotai_name_el[0]== "RNAi":
            kind = kotai_name_el[1]
        else:
            kind = kotai_name_el[0]
        return kind 

    @staticmethod
    def get_cellStage(
        h5_path: Union[ str | os.PathLike ],
        T : Union[ str | int ],
    ) -> str:
        # open BD5 in read mode and acccess cell stage via T, Z 
        # f: data/{TimePoint}/object/ 
        with  h5py.File(h5_path ,"r") as f:

            try:
                H5_dataset = f[f'data/{str(T)}/object/0']

                # e.g)
                # np.void(
                #    (b'76001', b'76', b'line', 11, b'366', b'354', b'28', b'AB'), 
                #    dtype={
                #       'names': ['ID', 't', 'entity', 'sID', 'x', 'y', 'z', 'label'], 
                #       'formats': ['S128', 'S32', 'S128', '<i4', 'S32', 'S32', 'S32', 'S128'], 
                #       'offsets': [0, 128, 160, 288, 292, 324, 356, 420], 
                #       'itemsize': 548i
                #   }
                # ),
                # via,
                # H5_dataset = list(H5_dataset)
                H5_dataset = np.array(H5_dataset)

                # 全部の輪郭座標の細胞名のラベル（np.byte_）
                # をsetで重複のない名前に集約して，utf-8に変換した．
                cell_names = set(
                    map(
                        lambda x: x.decode('UTF-8'), 
                        set(H5_dataset[:]['label'])
                    )
                )

            except KeyError as e:
                # logger.error(
                #     f"got error {e} and init no cell_name in {h5_path}\t{T}"
                # )
                cell_names = set()

            cell_stage = len(cell_names)

        return cell_stage,cell_names

    @staticmethod
    def fetch_Ts_from_cellStage(
        BD5_path, # 個体ごと
        in_dir :Union[str | os.PathLike],
        target_cellStage: int = 1, 
        Z_range :Sequence[int] = (35,40),
    ) -> Sequence[ str | os.PathLike ]:
        in_dir  = Path(in_dir)
        img_path_list = [p for p in in_dir.iterdir() if p.is_file()]

        cell_stage_img_paths = []

        with logging_redirect_tqdm(loggers=[logger]):
            for path in tqdm(img_path_list):

                [base_dir, filename, suffix] = WDDD2FileNameUtil.strip_path(path,)
                [kotai_kind, kotai_name, T, Z,] = WDDD2FileNameUtil.strip_name(filename, )

                if not int(Z) in Z_range:
                    continue
                cell_stage, cell_names = WDDD2FileNameUtil.get_cellStage(BD5_path, T,)
                if cell_stage != target_cellStage:
                    continue

                cell_stage_img_paths.append(path)
        
        return cell_stage_img_paths


    @staticmethod
    def convert_name_from_wt_to_pseudoAnomaly(wt_name, kind):
        e = wt_name.split("_")
        e[0] = "pseudoAnomaly"
        e[1] = kind
        pseudoAnomaly_name = "_".join(e)
        return pseudoAnomaly_name 



if __name__ == "__main__":

    logging.basicConfig(filename='./debug.log', filemode='w', level=logging.DEBUG)

    img_path = '/mnt/d/WDDD2/TIF_GRAY/wt_N2_081113_01/wt_N2_081113_01_T73_Z35.tiff',
    [base_dir, filename, suffix] = WDDD2FileNameUtil.strip_path(img_path)
    [kotai_kind, kotai_name, T, Z,] = WDDD2FileNameUtil.strip_name(filename)

    cell_stage ,cell_names = WDDD2FileNameUtil.get_cellStage(
        '/mnt/d/WDDD2/BD5/wt_N2_081113_01_bd5.h5', T
    ) 

    cell_stage_img_paths = WDDD2FileNameUtil.fetch_Ts_from_cellStage(
                                '/mnt/d/WDDD2/BD5/wt_N2_081113_01_bd5.h5', 
                                '/mnt/d/WDDD2/TIF_GRAY/',
                                target_cellStage = 1, 
                                Z_range= (35,40),
                            )


