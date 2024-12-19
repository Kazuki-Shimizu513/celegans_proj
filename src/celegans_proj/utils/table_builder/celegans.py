


import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


import os
from typing import Union
from collections.abc import Sequence

from pathlib import Path

from pandas import DataFrame
import pandas as pd 
from tabulate import tabulate
import numpy as np



from celegans_proj.utils.table_builder.base import BaseTableBuilder
from celegans_proj.utils.file_import import  WDDD2FileNameUtil

logger = logging.getLogger(__name__)



class WildTypeCelegansTableBuilder(BaseTableBuilder):
    def __init__(
        self,
        in_path: Union[ str | os.PathLike ] = Path('~/WDDD2/TIF_GRAY'), 
        out_path: Union[ str | os.PathLike ] = Path('~/WDDD2_AD/meta_data/datatable'), 
        split: bool = True , 
        split_ratio: Sequence[ Union[ float | int ] ] = [24, 1, 8],
        h5_path: Union[ str | os.PathLike ] = Path("~/WDDD2/BD5"),
        Z_range :tuple =  (33, 40),
        T_cellStage : Sequence[int] = (2, ),
        shuffle: bool = False,
        random_seed: int = 44,
    ) -> None:

        super().__init__(
            in_path, 
            out_path, 
            split, 
            split_ratio,
        )
        self.h5_path = h5_path
        self.Z_range = Z_range
        self.T_cellStage = T_cellStage

        self.shuffle = shuffle
        self.rs = np.random.RandomState(random_seed)


    def __call__(
        self,
    ) -> None:
        [df_train, df_val, df_test, df_all] = self.build_table()
        self.save_df2csv(df_all , "wildtype_all",)
        self.save_df2csv(df_train , "wildtype_train",)
        self.save_df2csv(df_val , "wildtype_val",)
        self.save_df2csv(df_test , "wildtype_test",)

    # TODO:: df_testに野生胚以外を追加でできるように改変する
    def build_table(
        self, 
    ) -> Sequence[DataFrame]:

        '''

            Build DataFrame and store it in ~/data/meta_data
            DataFrame Contains

             ---------------------------------------------------------------------------------------
            |   | kotai_name      | H5_path                | img_path                   | split
            =======================================================================================
            | 0 | wt_N2_081113_01 | wt_N2_081113_01_bd5.h5 | wt_N2_081113_01_T0_Z0.tiff | train 
             --------------------------------------------------------------------------------------

        '''

        # 1. get all (wildytpe) kotai_dir path 
        kotai_list = self.get_kotai_list()

        logger.debug(
            f'{kotai_list=}\n'
        )


        # 2. split with respect to kotai_num 
        (train, val, test) = self.split_items(
                                kotai_list,
                                shuffle = self.shuffle,
                                random_state =  self.rs, 
                             )
        logger.debug(
            f'{train=},\n{val=},\n{test=}\n\n'
        )


        # 3. get all img path within Z_range and T_cellStage
        train_img_list = self.get_img_path_list(train,split_name='train')
        val_img_list = self.get_img_path_list(val,split_name='val')
        test_img_list = self.get_img_path_list(test,split_name='test')


        # 4. store img_path into a DataFrame 
        col_name = [
            "kotai_name",
            "h5_path",
            "img_path",
            "split",
            "cell_stage",
            "T",
            "Z",
            "cell_names",
            "kotai_kind"
        ]
        df_train = DataFrame(train_img_list, columns=col_name)
        df_val = DataFrame(val_img_list, columns=col_name)
        df_test = DataFrame(test_img_list, columns=col_name)

        df_all =  self.concat_dfs(df_train, df_val, sort = True, value_title = ['kotai_name','T', 'Z',],)
        df_all =  self.concat_dfs(df_all, df_test, sort = True, value_title = ['kotai_name','T', 'Z',],)

        logger.debug(
            tabulate(df_all.head(), headers='keys', tablefmt='psql')
        )

        return [df_train, df_val, df_test, df_all]

    def get_kotai_list(
        self,
        only:str = ["WT"],

    ) -> Sequence[ str ]:
        in_path = Path(self.in_path)
        in_path = in_path.resolve()

        kotai_list = [str(p.name) for p in in_path.iterdir() if p.is_dir()] # path
        if "WT" in only :
            kotai_list = [name for name in kotai_list if 'wt_' in name] 
        elif "RNAi" in only:
            kotai_list = [name for name in kotai_list if 'RNAi_' in name] 
        else: #PseudoAnomaly
            kotai_list = [name for name in kotai_list if 'PseudoAnomaly_' in name] 

        return kotai_list

    def get_img_path_list(
        self,
        kotai_split_list: Sequence[ str ],
        split_name: str = "train", 
    ) -> Sequence[ str ]:
        logger.info(
            f"building {split_name} dataFrame"
        )
        in_path = Path(self.in_path)
        h5_path = Path(self.h5_path)

        kotai_path_list = [Path(in_path.joinpath(str(name)).resolve()) for name in kotai_split_list]
        kotai_h5_list = [Path(h5_path.joinpath(str(name)+"_bd5.h5").resolve()) for name in kotai_split_list]

        # H５をよんで，Z＿範囲とT_細胞期に入っている画像のみを抽出
        kotai_img_list = []
        with logging_redirect_tqdm(loggers=[logger]):
            for kotai_path, kotai_h5 in tqdm(zip(kotai_path_list, kotai_h5_list)):
                for p in tqdm(kotai_path.iterdir()):

                    # img file check
                    if not p.is_file():
                        continue

                    # Z check
                    [base_dir, filename, suffix] = WDDD2FileNameUtil.strip_path(p)
                    [kotai_kind, kotai_name, T, Z,] = WDDD2FileNameUtil.strip_name(filename)
                    if not (self.Z_range[0] <= int(Z) and self.Z_range[1] >= int(Z)):
                        continue

                    # T check
                    cell_stage,cell_names = WDDD2FileNameUtil.get_cellStage(kotai_h5,T,)
                    if not (self.T_cellStage[0] == int(cell_stage)):
                        continue
                    try: # TODO: 汚すぎるので直す．範囲がなくてもいいように変更する
                        if not (self.T_cellStage[1] >= int(cell_stage)):
                            continue
                    except:
                        pass


                    img_tuple= tuple([
                        str(kotai_name), 
                        str(kotai_h5), 
                        str(p.resolve()), 
                        str(split_name),
                        str(cell_stage),
                        str(T),
                        str(Z),
                        str(cell_names),
                        str(kotai_kind),
                    ])

                    kotai_img_list.append(img_tuple)

        return kotai_img_list


class RNAiCelegansTableBuilder(WildTypeCelegansTableBuilder):

    ''' C.elegans DataTable BUilder for RNAi one

    '''
    def __init__(
        self,
        in_path: Union[ str | os.PathLike ] = Path('~/WDDD2/TIF_GRAY'), 
        out_path: Union[ str | os.PathLike ] = Path('~/WDDD2_AD/meta_data/datatable'), 
        h5_path: Union[ str | os.PathLike ] = Path("~/WDDD2/BD5"),
        Z_range :tuple =  (33, 40),
        T_cellStage : Sequence[int] = (2, ),
        shuffle: bool = False,
        random_seed: int = 44,
    ) -> None:

        super().__init__(
            in_path=in_path, 
            out_path=out_path, 
            h5_path=h5_path,
            Z_range=Z_range,
            T_cellStage=T_cellStage,
            shuffle= shuffle,
            random_seed = random_seed,
        )

    def __call__(
        self,
    ) -> None:
        df_test = self.build_table()
        self.save_df2csv(df_test , "RNAi_test",)

    def build_table(
        self, 
    ) -> DataFrame:

        '''

            Build DataFrame and store it in ~/data/meta_data
            DataFrame Contains

             ---------------------------------------------------------------------------------------
            |   | kotai_name      | H5_path                | img_path                   | split
            =======================================================================================
            | 0 | RNAi_N2_081113_01 | RNAi_N2_081113_01_bd5.h5 | RNAi_N2_081113_01_T0_Z0.tiff | test 
             --------------------------------------------------------------------------------------

        '''

        # 1. get all (wildytpe) kotai_dir path 
        kotai_list = self.get_kotai_list(only=["RNAi"])

        logger.debug(
            f'{kotai_list=}\n'
        )

        test_img_list = self.get_img_path_list(kotai_list,split_name='test')


        # 4. store img_path into a DataFrame 
        col_name = [
            "kotai_name",
            "h5_path",
            "img_path",
            "split",
            "cell_stage",
            "T",
            "Z",
            "cell_names",
            "kotai_kind"
        ]
        df_test = DataFrame(test_img_list, columns=col_name)

        logger.debug(
            tabulate(df_test.head(), headers='keys', tablefmt='psql')
        )

        return  df_test
















