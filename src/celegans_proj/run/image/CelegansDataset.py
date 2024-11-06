import numpy as np
import pandas as  pd
import torch 
from pathlib import Path
from torchvision import transforms
from datasets import Image
from datasets import Dataset
from PIL import Image as PILImage

from myProject.utils.make_csv import (
    make_datatable,
    make_train_img_path_csv,
    make_test_img_path_csv,
    split_img_path_csv,
    augment_img_path_csv,
    make_pseudo_anomaly_test_img_path_csv,
    make_all_pseudo_anomaly_test_img_path_csv,
    get_name_of_dataset,
    make_specific_cell_img_path_csv,
    concat_dfs,
)


# TODO:: make CelegansDataset for 3D+T 
# class Celegans3dDataset(Dataset):
#     def __init__(self, args):
#         self.img_labels = pd.read_csv(args.datasets_file)

#     def __len__(self):
#         return len(self.img_urls)
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)

#         if self.transform:
#             image = self.transform(image)
#         return image, name
  
class DatasetPathGenerator:
    def __init__(self, args, mode='train'):

        ## INPUT
        self.mode = mode
        self.args = args
        input_dir = Path(args.input_dir1)
        base_path = input_dir.joinpath("WDDD2")
        self.data_bd5_path = base_path.joinpath("BD5")
        self.data_imgs_path = base_path.joinpath("TIF_GRAY")
        input_dir2 = Path(args.input_dir2)
        base_dir = input_dir2.joinpath("WDDD2")
        self.data_pseudo_anomaly_path = base_dir.joinpath("pseudo_anomaly_celegans")
        self.data_augmented_imgs_path = base_dir.joinpath("augmentation_celegans")

        ## CFGS
        self.min_Z = args.cell_z_1
        self.MAX_Z = args.cell_z_2

        ## MIDDLE
        base_dir2 = Path(args.output_dir)
        if self.args.debug_flag:
            self.out_dir_path = base_dir2.joinpath("datatable_debug")
        else:
            self.out_dir_path = base_dir2.joinpath("datatable")
        self.train_img_path = self.out_dir_path.joinpath("train_img_path.csv")
        self.data_table_path = self.out_dir_path.joinpath("datatable.csv")
        self.one_cell_test_set_path = self.out_dir_path.joinpath("oneCell_test_dataset.csv")

        self.out_dir_path.mkdir(parents=True, exist_ok=True)
        self.train_img_path.touch(exist_ok=True)
        self.data_table_path.touch(exist_ok=True)
        self.one_cell_test_set_path.touch(exist_ok=True)

        ## OUTPUT
        self.train_set_path = self.out_dir_path.joinpath("train_dataset.csv")
        self.augmented_train_dataset = self.out_dir_path.joinpath("augmented_train_dataset.csv")

        self.val_set_path = self.out_dir_path.joinpath("val_dataset.csv")

        self.test_set_path = self.out_dir_path.joinpath("test_dataset.csv")
        self.pseudo_test_set_path = self.out_dir_path.joinpath("pseudo_test_dataset.csv")
        self.RNAi_test_set_path = self.out_dir_path.joinpath("RNAi_test_dataset.csv")


        # TODO:: return test_set_path for 2 pattern 
        # TODO:: (wt_test+pseudo, wt_test+RNAi) 

        self.train_set_path.touch(exist_ok=True)
        self.augmented_train_dataset.touch(exist_ok=True)

        self.val_set_path.touch(exist_ok=True)

        self.test_set_path.touch(exist_ok=True)
        self.pseudo_test_set_path.touch(exist_ok=True)
        self.RNAi_test_set_path.touch(exist_ok=True)

    def __call__(self):
        self.generate_csvs()
        args = self.update_args()

        # 各データセットにたいして画像の名前を持ってきたい

        if self.mode == 'train':
            return (
                args,
                get_name_of_dataset(self.train_set_path),
                get_name_of_dataset(self.val_set_path),
                get_name_of_dataset(self.test_set_path),
            )
        else:
            return (
                args,
                get_name_of_dataset(self.test_set_path),
                get_name_of_dataset(self.pseudo_test_set_path),
                get_name_of_dataset(self.RNAi_test_set_path),
            )

    def update_args(self):
        
        self.args.train_img_path = self.train_img_path
        self.args.data_table_path = self.data_table_path
        self.args.train_set_path = self.train_set_path
        self.args.val_set_path = self.val_set_path

        self.args.test_set_path = self.test_set_path
        self.args.pseudo_test_set_path = self.pseudo_test_set_path
        self.args.RNAi_test_set_path = self.RNAi_test_set_path

        print("args updated")


        return self.args

    def generate_csvs(self):
        if self.args.prepare_datatable_flag:
            make_datatable(

                str(self.data_bd5_path), 
               str(self.data_imgs_path), 

               str(self.data_table_path), 
               self.args.debug_flag
            )

        if self.args.prepare_datasetCSV_flag:
            make_train_img_path_csv(str(self.data_table_path), 
                                    self.args.cell_stage, 
                                    self.min_Z, 
                                    self.MAX_Z, 
                                    self.args.debug_flag
            )

        if self.args.split_img_path_csv_flag:
            split_img_path_csv(
                str(self.train_img_path),
                out_dir_path=str(self.out_dir_path),
                train=0.75,
                val=0.1,
                test=0.15,
                debug=self.args.debug_flag,
            )

        self.add_augmented_images_to_train_set()

        self.add_pseudo_anomaly_images_to_test_set()
        self.add_RNAi_images_to_test_set()

        if self.args.debug_flag \
            and self.mode == 'train':
            print(f"{self.train_set_path=}")
            print(f"{self.val_set_path=}")
            print(f"{self.test_set_path=}")
        else:
            print(f"{self.test_set_path=}")
            print(f"{self.pseudo_test_set_path=}")
            print(f"{self.RNAi_test_set_path=}")

    def add_augmented_images_to_train_set(self):
        #  ADD Augmented Image Data to train_set_path
        if self.args.add_data_augumented_imgs_to_train_csv_flag:
            if self.args.debug_flag:
                print(f"{self.args.augmentation_modes=}")
            augment_img_path_csv(
                str(self.args.train_set_path),
                self.args.augmentation_modes,
                input_dir_path=str(self.data_augmented_imgs_path),
                out_dir_path=str(self.out_dir_path),
                debug=self.args.debug_flag,
            )

        if self.args.use_augmented_train_dataset:
            self.args.train_set_path = self.augmented_train_dataset 


    def add_pseudo_anomaly_images_to_test_set(self):
        #  ADD Pseudo Anomaly Image Data to test_set_path
        if self.args.add_pseudo_anomaly_imgs_to_test_csv_flag:
            if self.args.debug_flag:
                print(f"{self.args.pseudo_anomaly_modes=}\t{self.min_Z=},{self.MAX_Z=}")
            if "oneCell" in self.args.pseudo_anomaly_modes:
                self.args.pseudo_anomaly_modes.remove("oneCell")
                one_cell = True
                #TODO 同じ個体の１cell 追加する
                make_specific_cell_img_path_csv(
                    str(self.data_bd5_path), 
                    str(self.data_imgs_path), 
                    str(self.data_pseudo_anomaly_path),
                    1,# self.args.cell_stage,
                    self.min_Z,
                    self.MAX_Z,
                    "oneCell",
                    str(self.test_set_path),
                    out_img_path=str(self.one_cell_test_set_path),
                    debug=self.args.debug_flag,
                )
            else:
                one_cell = False 


            make_pseudo_anomaly_test_img_path_csv(
                str(self.test_set_path),
                self.args.pseudo_anomaly_modes,
                input_dir_path=str(self.data_pseudo_anomaly_path),
                out_dir_path=str(self.out_dir_path), # TODO:;
                debug=self.args.debug_flag,
            )

            if one_cell:
                concat_dfs(
                    str(self.out_dir_path) + "/pseudo_test_dataset.csv",
                    str(self.one_cell_test_set_path),
                    str(self.out_dir_path) + "/pseudo_test_dataset.csv",
                    sort=False,
                    debug=self.args.debug_flag,
                )

    def add_RNAi_images_to_test_set(self):
        if self.args.add_anomaly_imgs_to_test_csv_flag:
            if self.args.debug_flag:
                print(f"before\t{self.min_Z=},{self.MAX_Z=}")
                self.min_Z=35; self.MAX_Z=35;
                print(f"{self.args.anomaly_gene_list=}\t{self.min_Z=},{self.MAX_Z=}")

            make_test_img_path_csv(
                str(self.data_table_path),
                self.args.cell_stage,
                self.min_Z,
                self.MAX_Z,
                self.args.anomaly_gene_list,
                str(self.test_set_path),
                out_img_path=self.RNAi_test_set_path,
                debug=self.args.debug_flag,
            )


class YouTransform:
    """
        Transform a Image Brightness with below math eq. 
        The amount of change in brightness value of pixels 
        close to the average value is small, and the amount 
        of change in brightness value of bright pixels 
        far from the average value is large.

        I_out = \arctan(\frac{ I_ori - I_mean }{ I_mean }) * alpha + I_mean 

    Args:
        Image (0-255 rank):

        alpha (float): 
            much bigger value leads more transform
    Returns:
        Transformed Image  
    Refs:
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, mean_list, alpha=50) -> None:
        self.alpha = alpha
        self.mean_list = mean_list


    def __call__(self, x, idx):
        # print(x.dtype) # torch.float32
        # PIL.Image -> np
        # x = np.array(x)
        # torch -> np
        x = x.to('cpu').detach().numpy().copy()

        # You Transform with numpy
        I_mean = self._get_image_mean(x) if self.mean_list[idx] is None else self.mean_list[idx]
        x = np.arctan((x - I_mean) / I_mean) * self.alpha + I_mean 

        # np -> PIL.Image
        # x = PILImage.fromarray(np.float32(x))
        # np -> torch.float32
        x = torch.from_numpy(x.astype(np.float32)).clone()
        return x


    def _get_image_mean(self, img):
        return np.mean(img)

    @staticmethod
    def get_image_mean(img_paths):
        img_list = [np.array(PILImage.open(path)) for path in img_paths]
        mean_list = list()
        for img in img_list:
            mean = np.mean(img)
            mean_list.append(mean)
        return mean_list
    @classmethod
    def get_volume_mean(cls, img_paths):
        name_list = [cls.get_img_name(path) for path in img_paths]
        img_list = [np.array(PILImage.open(path)) for path in img_paths]
        vol_list = list()
        vol_tmp = list()
        vol_idxs = list()
        name_tmp = None
        for idx, (img, name) in enumerate(zip(img_list, name_list)):
            kotai_name, t_idx, z_idx = cls.split_name_element(name)

            if name_tmp is None: # first case
                name_tmp = kotai_name

            if kotai_name == name_tmp:
                vol_tmp.append(img)
            else:
                vol_list.append(vol_tmp)
                vol_tmp = list()
                name_tmp = kotai_name
                vol_tmp.append(img)

            vol_idxs.append(len(vol_list))

            if idx == len(img_paths)-1: # last case
                vol_list.append(vol_tmp)
                # print("last case")


            assert idx == len(vol_idxs)-1,\
                f"{idx=}\t{len(vol_idxs)-1=}\t{len(vol_list)=}\t{kotai_name=}"


            # print(f"{len(vol_list)=}\t{kotai_name=}") #23
        # print(f"{len(vol_idxs)=}") #919
        # print(f"{vol_idxs=}")

        assert len(vol_idxs)==len(img_paths),\
            f"{len(mean_list)=}\t{len(img_paths)=}"

        mean_list = list()
        prev_idx = None
        for idx in vol_idxs:
            # print(f"{idx=}\t{len(vol_list)=}\t{len(vol_list[idx])=}")
            mean = prev_mean if prev_idx==idx else np.mean(vol_list[idx])/255.0
            mean_list.append(mean)
            prev_idx = idx 
            prev_mean = mean

        assert len(mean_list)==len(img_paths),\
            f"{len(mean_list)=}\t{len(img_paths)=}"

        return mean_list


    def get_spacioTemporal_mean(self,img_paths):
        # TODO:: 個体，時間，深さごとでvolumeの平均（のリスト）を算出したい
        return mean

    @staticmethod
    def get_img_name(path):
        # /mnt/d/WDDD2/TIF_GRAY/wt_N2_081113_01/wt_N2_081113_01_T47_Z35.tiff
        # /mnt/d/WDDD2/TIF_GRAY/RNAi_F10E9.8_100602_04/RNAi_F10E9.8_100602_04_T38_Z33.tiff
        # wt_N2_081007_01_move_T74_Z35.tiff

        path = str(path)
        name_el = path.split("/")[-1]
        name_el = name_el.split(".")[:-1]
        if len(name_el) > 1:
            name = '.'.join(name_el)
        else:
            name = name_el[0]
        return name

    @staticmethod
    def split_name_element(name):
        name_el = name.split("_")
        z_idx = int(name_el[-1][1:])
        t_idx = int(name_el[-2][1:])
        kotai_name = "_".join(name_el[:-2])
        # print(f"{z_idx=}, {t_idx=}, {kotai_name=}")
        return kotai_name, t_idx, z_idx


class myCompose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Refs: 
        https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Compose

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        #     _log_api_usage_once(self)
        self.transforms = transforms

    def __call__(self, img, idx):
        for t in self.transforms:
            if t ==  self.transforms[-1]:
                img = t(img, idx)
            else:
                img = t(img)

        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

def transform_images(examples):
    augmentations = myCompose(
        [
            transforms.Resize(
                (256, 256),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(), # [0,255]->[0,1]
            YouTransform(examples["mean"], alpha=0.2, ), # 50/255
        ]
    )
    images = [augmentations(image, idx) for idx, image in enumerate(examples["image"])]
    return {"input": images}


#     # TODO:: add return img_name 
#     names = [name for name in examples["name"]]
#     return {"input": images, "name":names}


def get_ds_dataloader(img_set_path, batch_size=1, shuffle=True, workers=1):
    # print(f"{img_set_path=}{batch_size=}")
    img_list = pd.read_csv(img_set_path)["img_path"].unique()
    name_list = [YouTransform.get_img_name(path) for path in img_list]

    print("get mean")

    # TODO: store volime mean in datatable path 
    mean_list = YouTransform.get_volume_mean(img_list)
    # mean_list = [None] * len(img_list) # YouTransform.get_image_mean(img_list)
    print("get mean finished")

    # dataset  の作成
    # TODO :: conbart to torch.Dataset class obj.
    dataset = Dataset.from_dict({"image": img_list}
                                ).cast_column(
                                "image", Image()
                                )
    dataset = dataset.add_column(name="name", column=name_list)
    dataset = dataset.add_column(name="mean", column=mean_list)
    dataset.set_transform(transform_images)

    # dataloader の作成
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
    )
    return dataset, dataloader


def get_2d_ds_dataloader(args):
    return dataset, dataloader
 

if __name__ == "__main__":
    print("this is CelegansDataset.py")



#     # Testing YouTransform
#     from PIL import Image, ImageFilter
#     from myProject.utils.visualize_results import plot_2img

    out_dir = Path("/mnt/c/Users/compbio/Desktop/shimizudata")

#     img_path = "/mnt/d/WDDD2/TIF_GRAY/wt_N2_081113_01/wt_N2_081113_01_T50_Z50.tiff"

#     img_ori = Image.open(img_path)
#     trsfrm = YouTransform(50)
#     img_out = trsfrm(img_ori)
#     img_ori_np = np.asarray(img_ori)
#     img_out_np = np.asarray(img_out)
#     print(f"{img_ori_np.max()=}{img_ori_np.min()=}{img_out_np.max()=}{img_out_np.min()=}")
#     plot_2img(img_ori_np, img_out_np,
#               "YouTransform_compare", "original", "transformed", 
#               out_dir, fontsize=24, vmax=255, vmin=0)

    # testing get_ds_dataloader
    from myProject.parser import parse_args
    from myProject.utils.parser_utils import (
        load_yaml_configs, update_args,
    )
    from myProject.utils.visualize_results import check_imgs 

    args = parse_args()
    cfgs = load_yaml_configs(args.yaml_path)
    args = update_args(args, cfgs)
    print(args)


    dataset_generator = DatasetPathGenerator(args)
    (args,
    train_img_name_list,
    val_img_name_list,
    test_img_name_list,) = dataset_generator()

    train_dataset, train_dataloader = get_ds_dataloader(
                            args.train_set_path, 
                            batch_size=args.train_batch_size, 
                            shuffle=False,
                            workers=args.dataloader_num_workers,
                            )

    validation_dataset, validation_dataloader = get_ds_dataloader(
                            args.val_set_path, 
                            batch_size=args.train_batch_size, 
                            shuffle=False,
                            workers=args.dataloader_num_workers,
                            )


    test_dataset, test_dataloader = get_ds_dataloader(
                            args.test_set_path, 
                            batch_size=args.test_batch_size, 
                            shuffle=False,
                            workers=args.dataloader_num_workers,
                            )

    check_imgs(train_dataloader, str(out_dir.joinpath("a")))
    check_imgs(validation_dataloader, str(out_dir.joinpath("b")))
    check_imgs(test_dataloader, str(out_dir.joinpath("c")))




    print("finish")

