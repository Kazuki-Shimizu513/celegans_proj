
import os
from abc import ABC
from typing import Union
from collections.abc import Sequence

from pathlib import Path

from pandas import DataFrame
import pandas as pd 


from sklearn.model_selection import train_test_split


class BaseTableBuilder(ABC):
    def __init__(
        self, 
        in_path: Union[ str | os.PathLike ] = Path('./data'), 
        out_path: Union[ str | os.PathLike ] = Path('./data'), 
        split: bool = True , 
        split_ratio: Union[ float | Sequence ] = [0.6, 0.1, 0.3],
    ):
        self.in_path = in_path
        self.out_path = out_path
        self.split = split
        self.split_ratio = split_ratio

    def sort_items(
        self, 
        df: DataFrame,
        value_title: Union[ str | Sequence ],
    ) -> DataFrame:
        return df.sort_values(value_title)

    def split_items(
        self, 
        item_list,
        shuffle : bool = False,
        random_state : int = 44, 
    ) -> Sequence[DataFrame]:

        train_val, test = train_test_split(
            item_list, 
            test_size=self.split_ratio[-1], 
            shuffle = shuffle,
            random_state = random_state, 
        ) 
        train, val = train_test_split(
            train_val, 
            test_size=self.split_ratio[-2], 
            shuffle = shuffle,
            random_state = random_state, 
        ) 
        return (train, val, test)

    def concat_dfs(
        self,
        df_1: DataFrame,
        df_2: DataFrame,
        sort: bool = True,
        value_title: Union[ str | Sequence[str] ] = 'ID',
    ) -> DataFrame:
        df_new = pd.concat([df_1, df_2], axis=0, )
        if sort:
            df_new = self.sort_items(df_new, value_title=value_title,)
        return df_new 

    def save_df2csv(
        self, 
        df : DataFrame, 
        file_name: Union[ str | os.PathLike ], 
        index: Union[bool | str] = False,
    ) -> None:
        context = Path(self.out_path)
        file_name = str(file_name) + ".csv"

        df.to_csv(
            str(context.joinpath(file_name)),
            index = index, 
        )


