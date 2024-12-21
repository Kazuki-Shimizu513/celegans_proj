


import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import itertools

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

# import cv2 as cv
# from PIL import Image
# import albumentations as A

# from sklearn.model_selection import train_test_split

# import tifffile
# import h5py

from celegans_proj.utils.table_builder.celegans import (
    WildTypeCelegansTableBuilder, 
    RNAiCelegansTableBuilder,
)

logger = logging.getLogger(__name__)



class Transporter:
    def __init__(
        self,

        transport: Sequence[str] = ["WT", "RNAi", "PseudoAnomaly", ],
        in_path: Union[ str | os.PathLike ] = Path('./WDDD2/TIF_GRAY'), 
        out_path: Union[ str | os.PathLike ] = Path('~/WDDD2_AD'), 
        split: bool = True , 
        split_ratio: Union[ float | Sequence ] = [24, 1, 8],
        h5_path: Union[ str | os.PathLike ] = Path("./WDDD2/BD5"),
        orf_list: Sequence[str] =["C53D5.6", "C29E4.8"],
        Z_range :tuple =  (33, 40),
        T_cellStage : Sequence[int] = (2, ),
        shuffle: bool = False,
        random_seed: int = 44,
   
    ) -> None:
        self.transport  = transport

        self.in_path = in_path
        self.out_path = out_path
        self.h5_path = h5_path
        self.datatable_path = out_path.joinpath('meta_data/datatable')
        # self.kinds = kinds
        # self.labels = labels
        # self.splits = splits


        self.wt_table_builder = WildTypeCelegansTableBuilder(
                                    in_path,
                                    self.datatable_path,
                                    split,
                                    split_ratio,
                                    h5_path,
                                    Z_range,
                                    T_cellStage,  
                                    shuffle,
                                    random_seed,
                                )

        self.RNAi_table_builder = RNAiCelegansTableBuilder(
                                    in_path,
                                    self.datatable_path,
                                    h5_path,
                                    orf_list,
                                    Z_range,
                                    T_cellStage,  
                                    shuffle,
                                    random_seed,
                                  )


    def __call__(
            self, 
    )->None:

        # 1. buld destination dir structure  
        self.build_WDDD2_AD_dir_structure(self.out_path)
        datatable_paths = [p.resolve() for p in self.datatable_path.iterdir() if p.is_file()]

        if "WT" in self.transport:
            trans_kind =  "wildtype_"
            # 2. buld DataFrame from in_dir via WildTypeCelegansTableBuilder
            wt_paths = [p.resolve() for p in datatable_paths if (trans_kind  in p.stem)]
            # if  len(wt_paths) == 0 :
            self.wt_table_builder()

            # 3. transfer (copy) to WDDD2_AD dataset
            self.transfer(
                wt_paths,
                trans_kind = trans_kind,
            )

        if "RNAi" in self.transport :
            trans_kind = "RNAi_"
            # 2. buld DataFrame from in_dir via RNAiCelegansTableBuilder
            RNAi_paths = [p.resolve() for p in datatable_paths if (trans_kind in p.stem)]
            # if  len(RNAi_paths) == 0 :
            self.RNAi_table_builder()
            # 3. transfer (copy) to WDDD2_AD dataset
            self.transfer(
                RNAi_paths,
                trans_kind = trans_kind,
            )
        if "pseudoAnomaly" in self.transport:
            trains_kind = "pseudoAnomaly_"

            datatable_path = self.datatable_path.joinpath('wildType_test.csv')
            generator = PseudoAnomalyGenerator(
                datatable_path, 
                out_path= out_path,
                in_dir= in_path, 
            )
            generator()

    @staticmethod
    def build_WDDD2_AD_dir_structure(
        out_path: Union[str | os.PathLike],
        kinds: Sequence[str] = ["wildType",],
        labels: Sequence[str] = ['good', 'anomaly',],
        splits: Sequence[str] = ['train', 'val', 'test', 'ground_truth',],
    ) -> None:
        '''example dir structure)

          WDDD2_AD
          ├── wildType
          │   ├── (defects_config.json)
          │   ├── ground_truth
          │   │   └── good
          │   │       └── wt_N2_081007_02_T53_Z36_mask.tiff
          │   ├── test # (empty)
          │   ├── train
          │   │   └── good
          │   │       └── wt_N2_081007_02_T53_Z36.tiff
          │   └── validation
          ├── sas-4
          │   ├── (defects_config.json)
          │   ├── ground_truth
          │   │   └── anomaly
          │   ├── test
          │   │   └── anomaly 
          │   ├── train # (empty)
          │   └── validation # (empty)
          ├── par-3 
          ├── ...
          ├── shrink 
          └── meta-data
              ├── BD5
              │   └── wt_n2_081007_02_BD5.h5
              ├── datatable
              └── wddd2_gene.csv

        e.g)
        for image path
            ./data/WDDD2_AD/wildType/train/good/wt_n2_081007_02_T53_Z36.tiff
        for mask path
            ./data/WDDD2_AD/wildType/ground_truth/train/good/wt_N2_081007_02_T53_Z36_mask.tiff

        '''

        out_path = Path(out_path)
        out_path.resolve()
        meta_path = out_path.joinpath("meta_data")
        BD5_path = meta_path.joinpath("BD5")
        table_path = meta_path.joinpath("datatable")
        BD5_path.mkdir(parents=True, exist_ok=True)
        table_path.mkdir(parents=True, exist_ok=True)
        for kind, split, label in itertools.product(kinds, splits, labels):
            p = out_path / kind / split / label
            p.mkdir(parents=True, exist_ok=True)


    def transfer(
        self,
        df_paths: Sequence[ str | os.PathLike ],
        trans_kind = "wildType_",
    ) -> None:
        ''' Transfer imgs and BD5 
        and create mask_img (which is zeros) when it does not exist
        '''
        out_path = Path(self.out_path)
        out_path.resolve()

        datatable_path = Path(self.datatable_path)
        BD5_path = datatable_path.parent.joinpath("BD5")
        BD5_path.resolve()

        with logging_redirect_tqdm(loggers=[logger]):
            for df_path in tqdm(df_paths):
                if str(df_path.suffix) != '.csv':
                    continue
                print(df_path)
                df = pd.read_csv(df_path)

                logger.debug(
                    tabulate(df.head(), headers='keys', tablefmt='psql')
                )

                # copy H5 files
                try:
                    h5_list = df['h5_path'].unique()
                    self.copy_files_to_dir(h5_list,BD5_path,) 
                except:
                    pass

                # making the out path 
                kinds = df['kotai_kind'].unique() # 'wildType'
                self.build_WDDD2_AD_dir_structure(self.out_path, kinds = kinds,)
                split = df['split'].unique()[0]

                for kind in kinds:
                    ## for wildtype, label is always "good"
                    if kind == "wildType" or kind == "wt":
                        label = "good"
                        _kind = "wildType"
                    else:
                        label = "anomaly"
                        _kind = kind 
                    out_dir = out_path / _kind / split / label
                    # copy image files
                    img_list = df[df['kotai_kind'].isin([f"{kind}"])]['img_path'].to_list()
                    self.copy_files_to_dir(img_list,out_dir,) 

                # TODO: implement this block
                # copy mask files
                # out_dir = out_path / kind / 'ground_truth' / label
                # img_list = df['mask_path'].to_list()
                # self.copy_files_to_dir(img_list,out_dir,) 




    def copy_files_to_dir(
        self,
        path_list: Sequence[ str | os.PathLike ],
        out_dir: Union[ str | os.PathLike ],
        # suffix : str = '.tiff'
    ) -> None:

        out_dir = Path(out_dir)
        for path in tqdm(path_list):
            path = Path(path)
            copy(str(path), str(out_dir.joinpath(path.name)))

            # im = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            # img_path = Path(img_path)
            # img_name = img_path.stem
            # cv.imwrite(str(out_dir.joinpath(img_name+suffix)), im)



if __name__ == "__main__":

    logging.basicConfig(filename='./logs/debug.log', filemode='w', level=logging.DEBUG)

    # img_path = '/mnt/d/WDDD2/TIF_GRAY/wt_N2_081113_01/wt_N2_081113_01_T73_Z35.tiff',
    # [base_dir, filename, suffix] = WDDD2FileNameUtil.strip_path(img_path)
    # [kotai_kind, kotai_name, T, Z,] = WDDD2FileNameUtil.strip_name(filename)

    # cell_stage ,cell_names = WDDD2FileNameUtil.get_cellStage(
    #     '/mnt/d/WDDD2/BD5/wt_N2_081113_01_bd5.h5', T
    # ) 

    # in_data_path = Path('/mnt/d/WDDD2')
    # img_data_path = in_data_path.joinpath('TIF_GRAY')
    # h5_data_path = in_data_path.joinpath('BD5')
    # out_data_path = Path('/mnt/e/WDDD2_AD')

    in_data_path = Path('/home/skazuki/data/WDDD2')
    in_data_path = Path('/home/skazuki/data/WDDD2_tmp')
    img_data_path = in_data_path.joinpath('TIFF')
    h5_data_path = in_data_path.joinpath('BD5')
    out_data_path = Path('/home/skazuki/data/WDDD2_AD')


    transporter = Transporter(
        transport = ["RNAi"],
        # transport = ["WT"],
        in_path = img_data_path, 
        out_path = out_data_path, 
        split = True , 
        split_ratio = [24, 1, 8], # only for wt
        h5_path = h5_data_path,
        orf_list =["C53D5.a", "C29E4.8"], # imb-3, let-754
        Z_range  =  (35, 35),
        # Z_range  =  (33, 40),
        T_cellStage  = (2, ),
        shuffle = False,
        random_seed = 44,
    )
    transporter()

