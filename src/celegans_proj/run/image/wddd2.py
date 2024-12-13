
"""WDDD2 Dataset (CC BY-NC-SA 4.0).

Description:
-  Our project aims to obtain a collection of quantitative information about cell division dynamics in early Caenorhabditis elegans embryos when each of all essential embryonic genes in all essential genes is silenced individually by RNA interference (RNAi).

- The information is obtained by combining four-dimensional differential interference contrast (DIC) microscopy and computer image processing (Hamahashi et al. 2007). The information collection provides novel opportunities for developing quantitative and computational approaches towards understanding animal development.

- This database provides the collection of quantitative data and microscopy image data of nuclear division dynamics in early C. elegans embryos, and the results of computational phenotype analysis.

- The collection includes quantitative data and image data from 33 wild-type and 1142 RNAi-treated embryos for 263 genes, and additional image data exhibiting severe abnormal phenotypes. It also includes the results of embryonic lethal test for 350 essential embryonic genes.

- You can search data and results using Chromosome and Locus/ORFs,
- download the quantitateive data of wild-type and RNAi-treated embryos in BD5 data format, and differential interference contrast (DIC) microscopy images in Leica Image format (LIF). You can also view quantitative data and the corresponding microscopy images.

License:

References:
    - https://wddd.riken.jp/ 
"""

import logging
from collections.abc import Sequence
from pathlib import Path
import math
from PIL import Image

import numpy as np
import pandas as pd 
from pandas import DataFrame

import torch 
from torchvision.transforms.v2 import Transform
from torchvision.transforms import v2

from anomalib import TaskType
from anomalib.data.base import AnomalibDataset,AnomalibDataModule
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import (
    DownloadInfo,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    validate_path,
)

logger = logging.getLogger(__name__)


IMG_EXTENSIONS = (".tif", ".tiff")

# DOWNLOAD_INFO = DownloadInfo(
#     name="WDDD2",
#     url = "https://wddd.riken.jp/LIF/wt_N2_080930_02.lif"
#     hashsum="not provided",
# )

# CATEGORIES = (
#     'wildType',
#     'RNAi',
#     'pseudoAnomaly',
# )



class WDDD2_AD(AnomalibDataModule):
    """WDDD2_AD Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/MVTec"``.
        category (str): Category of the MVTec dataset (e.g. "bottle" or "cable").
            Defaults to ``"bottle"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
            Defaults to ``TaskType.SEGMENTATION``.
        image_size (tuple[int, int], optional): Size to which input images should be resized.
            Defaults to ``None``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        train_transform (Transform, optional): Transforms that should be applied to the input images during training.
            Defaults to ``None``.
        eval_transform (Transform, optional): Transforms that should be applied to the input images during evaluation.
            Defaults to ``None``.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defualts to ``None``.

    Examples:
        To create an MVTec AD datamodule with default settings:

        >>> datamodule = MVTec()
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

        >>> data["image"].shape
        torch.Size([32, 3, 256, 256])

        To change the category of the dataset:

        >>> datamodule = MVTec(category="cable")

        To change the image and batch size:

        >>> datamodule = MVTec(image_size=(512, 512), train_batch_size=16, eval_batch_size=8)

        MVTec AD dataset does not provide a validation set. If you would like
        to use a separate validation set, you can use the ``val_split_mode`` and
        ``val_split_ratio`` arguments to create a validation set.

        >>> datamodule = MVTec(val_split_mode=ValSplitMode.FROM_TEST, val_split_ratio=0.1)

        This will subsample the test set by 10% and use it as the validation set.
        If you would like to create a validation set synthetically that would
        not change the test set, you can use the ``ValSplitMode.SYNTHETIC`` option.

        >>> datamodule = MVTec(val_split_mode=ValSplitMode.SYNTHETIC, val_split_ratio=0.2)

    """

    def __init__(
        self,
        root: Path | str = "/mnt/e/WDDD2_AD",
        category: str = "wildType",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType | str = TaskType.CLASSIFICATION,#  SEGMENTATION,
        image_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
        debug:bool = False, 
        debug_data_ratio:int = 0.1, 
        add_anomalous: bool = False, 
 
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            image_size=image_size,
            transform=transform,
            train_transform=train_transform,
            eval_transform=eval_transform,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.task = TaskType(task)
        self.root = Path(root)
        self.category = category

        self.debug =debug
        self.debug_data_ratio = debug_data_ratio
        self.add_anomalous = add_anomalous 

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The stage argument is not used here. This is because, for a given instance of an AnomalibDataModule
            subclass, all three subsets are created at the first call of setup(). This is to accommodate the subset
            splitting behaviour of anomaly tasks, where the validation set is usually extracted from the test set, and
            the test set must therefore be created as early as the `fit` stage.

        """
        self.train_data = WDDD2_AD_DS(
            task=self.task,
            transform=self.train_transform,
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
            add_anomalous = self.add_anomalous , 
        )

        self.val_data = WDDD2_AD_DS(
            task=self.task,
            transform=self.eval_transform,
            split=Split.VAL,
            root=self.root,
            category=self.category,
            add_anomalous = self.add_anomalous , 
        )

        self.test_data = WDDD2_AD_DS(
            task=self.task,
            transform=self.eval_transform,
            split=Split.TEST,
            root=self.root,
            category=self.category,
            add_anomalous = self.add_anomalous , 
        )

        if self.debug:
            for attribute in ("train_data", "val_data", "test_data"):
                dataset = getattr(self, attribute)
                delattr(self, attribute)
                subset = self.random_subsampling(
                        dataset,
                        seed = self.seed,
                        ratio =self.debug_data_ratio,
                 )

                if attribute in ("train_data", "val_data",):
                    subset.samples = _add_anomalous(subset.samples, length=3)

                setattr(self, attribute, subset)

    def random_subsampling(
        self,
        dataset,
        seed = 44,
        ratio =0.1,
    ):
        subset :list[AnomalibDataset] = list()
        subset_length = math.floor(len(dataset.samples) * ratio)
        # perform random subsampling
        random_state = torch.Generator().manual_seed(seed) if seed else None
        indices = torch.randperm(len(dataset.samples), generator=random_state)
        subset_indices = torch.split(indices, subset_length)
        subset =  dataset.subsample(subset_indices[0])
        # print(f"total iterations: {len(subset)=}")
        return subset

#     def prepare_data(self) -> None:
#         """Download the dataset if not available.

#         This method checks if the specified dataset is available in the file system.
#         If not, it downloads and extracts the dataset into the appropriate directory.

#         Example:
#             Assume the dataset is not available on the file system.
#             Here's how the directory structure looks before and after calling the
#             `prepare_data` method:

#             Before:

#             .. code-block:: bash

#                 $ tree datasets
#                 datasets
#                 ├── dataset1
#                 └── dataset2

#             Calling the method:

#             .. code-block:: python

#                 >> datamodule = MVTec(root="./datasets/MVTec", category="bottle")
#                 >> datamodule.prepare_data()

#             After:

#             .. code-block:: bash

#                 $ tree datasets
#                 datasets
#                 ├── dataset1
#                 ├── dataset2
#                 └── MVTec
#                     ├── bottle
#                     ├── ...
#                     └── zipper
#         """
#         if (self.root / self.category).is_dir():
#             logger.info("Found the dataset.")
#         else:
#             download_and_extract(self.root, DOWNLOAD_INFO)


class WDDD2_AD_DS(AnomalibDataset):
    """WDDD2 dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``.
        root (Path | str): Path to the root of the dataset.
            Defaults to ``./datasets/MVTec``.
        category (str): Sub-category of the dataset, e.g. 'bottle'
            Defaults to ``bottle``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
            Defaults to ``None``.


    Examples:
        .. code-block:: python

            from caenorhabditiselegans-anomalydetection-dataset import WDDD2_AD 
            from anomalib.data.utils.transforms import get_transforms

            transform = get_transforms(image_size=256)
            dataset = WDDD2_AD(
                task="classification",
                transform=transform,
                root='./caenorhabditiselegans-anomalydetection-dataset/data',
                category= 'oneCell',
            )
            dataset.setup()
            print(dataset[0].keys())
            # Output: dict_keys(['image_path', 'label', 'image'])

        When the task is segmentation, the dataset will also contain the mask:

        .. code-block:: python

            dataset.task = "segmentation"
            dataset.setup()
            print(dataset[0].keys())
            # Output: dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

        The image is a torch tensor of shape (C, H, W) and the mask is a torch tensor of shape (H, W).

        .. code-block:: python

            print(dataset[0]["image"].shape, dataset[0]["mask"].shape)
            # Output: (torch.Size([3, 256, 256]), torch.Size([256, 256]))

    """

    def __init__(
        self,
        task: TaskType,
        root: Path | str = "./data",
        category: str = "oneCell",
        transform: Transform | None = None,
        split: str | Split | None = None,
        add_anomalous: bool = False, 
    ) -> None:
        super().__init__(task=task, transform=transform)

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples = make_WDDD2_dataset(self.root_category, split=self.split, extensions=IMG_EXTENSIONS, add_anomalous = add_anomalous, )
    

    def setup(self,):
        pass


def make_WDDD2_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
    add_anomalous: bool = False, 
) -> DataFrame:
    """Create WDDD2 AD samples by parsing the WDDD2 AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:

    +---+---------------+-------+---------+---------------+---------------------------------------+-------------+
    |   | path          | split | label   | image_path    | mask_path                             | label_index |
    +===+===============+=======+=========+===============+=======================================+=============+
    | 0 | datasets/name | test  | defect  | filename.png  | ground_truth/defect/filename_mask.png | 1           |
    +---+---------------+-------+---------+---------------+---------------------------------------+-------------+

    Args:
        root (Path): Path to dataset
        split (str | Split | None, optional): Dataset split (ie., either train or test).
            Defaults to ``None``.
        extensions (Sequence[str] | None, optional): List of file extensions to be included in the dataset.
            Defaults to ``None``.

    Examples:
        The following example shows how to get training samples from WDDD2 AD OneCell category:

        >>> root = Path('./WDDD2')
        >>> category = 'oneCell'
        >>> path = root / category
        >>> path
        PosixPath('WDDD2/oneCell')

        >>> samples = make_WDDD2_dataset(path, split='train', split_ratio=0.1, seed=0)
        >>> samples.head()
           path         split label image_path                       mask_path                                    label_index
        0  WDDD2/oneCell train good WDDD2/oneCell/train/good/105.png WDDD2/oneCell/ground_truth/good/105_mask.png 0
        1  WDDD2/oneCell train good WDDD2/oneCell/train/good/1.png WDDD2/oneCell/ground_truth/good/1_mask.png 0
        2  WDDD2/oneCell train good WDDD2/oneCell/train/good/5.png WDDD2/oneCell/ground_truth/good/5_mask.png 0
        3  WDDD2/oneCell train good WDDD2/oneCell/train/good/0.png WDDD2/oneCell/ground_truth/good/0_mask.png 0
        4  WDDD2/oneCell train good WDDD2/oneCell/train/good/15.png WDDD2/oneCell/ground_truth/good/15_mask.png 0

    Returns:
        DataFrame: an output dataframe containing the samples of the dataset.
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    samples_list = [(str(root),) + f.parts[-3:] for f in root.glob(r"**/*") if f.suffix in extensions]
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # separate masks from samples
    mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(by="image_path", ignore_index=True)
    samples = samples[samples.split != "ground_truth"].sort_values(by="image_path", ignore_index=True)

    # assign mask paths to anomalous test images
    samples["mask_path"] = ""
    samples.loc[
        (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
        "mask_path",
    ] = mask_samples.image_path.to_numpy()

    # assert that the right mask files are associated with the right test images
    abnormal_samples = samples.loc[samples.label_index == LabelName.ABNORMAL]
    if (
        len(abnormal_samples)
        and not abnormal_samples.apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1).all()
    ):
        msg = """Mismatch between anomalous images and ground truth masks. Make sure t
        he mask files in 'ground_truth' folder follow the same naming convention as the
        anomalous images in the dataset (e.g. image: '000.png', mask: '000.png' or '000_mask.png')."""
        raise MisMatchError(msg)

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    if add_anomalous:
        samples = _add_anomalous(samples, length=10)

    return samples

def gen_anomalous(img_path, msk_path, size=(600,600,),):
    img = Image.fromarray(np.ones(size, dtype=np.uint8))
    msk = Image.fromarray(np.ones(size, dtype=np.uint8))
    img.save(img_path)
    msk.save(msk_path)


def _add_anomalous(samples: DataFrame, length: int=10) -> DataFrame:
    """ add some anomalous samples for AUROC or F1 Metric CallBack

    dict_keys(['image_path', 'label', 'image', 'mask'])
    type(batch['image_path'])=<class 'str'>
        batch['image_path']='/mnt/e/WDDD2_AD/wildType/train/good/wt_N2_081007_02_T42_Z3
    3.tiff'
    type(batch['label'])=<class 'numpy.int64'>
        batch['label']=0
    type(batch['image'])=<class 'torchvision.tv_tensors._image.Image'>
        batch['image'].shape=torch.Size([1, 256, 256])
    type(batch['mask'])=<class 'torchvision.tv_tensors._mask.Mask'>
        batch['mask'].shape=torch.Size([256, 256])

    print(f"{samples.iloc[0].map(type)=}") 
    samples.iloc[0].map(type)=
        path                   <class 'str'>
        split                  <class 'str'>
        label                  <class 'str'>
        image_path             <class 'str'>
        label_index    <class 'numpy.int64'>
        mask_path              <class 'str'>
        Name: 0, dtype: object

    print(f"{samples.iloc[0]=}") 
    samples.iloc[0]=
        path                                    /mnt/e/WDDD2_AD/wildType
        split                                                      train
        label                                                       good
        image_path     /mnt/e/WDDD2_AD/wildType/train/good/wt_N2_0810...
        label_index                                                    0
        mask_path
        Name: 0, dtype: object
    """
    # print(f"{len(samples)=}")
    anomalous_list = []
    for l in range(length):
        img_path = f"{samples.iloc[len(samples)-1, 0,]}/{samples.iloc[len(samples)-1, 1,]}/anomaly/{l:0=3}.png"
        msk_path = f"{samples.iloc[len(samples)-1, 0,]}/ground_truth/anomaly/{l:0=3}_mask.png"
        inner_list = [
            samples.iloc[len(samples)-1, 0,], # "path"],
            samples.iloc[len(samples)-1, 1,], # "split"],
            "anomaly", # LabelName.ABNORMAL,
            img_path,
            np.int64(1),
            msk_path,
        ]
        anomalous_list.append(inner_list)
    anom_samples = DataFrame(anomalous_list, columns=samples.columns.values.tolist())
    samples = pd.concat([samples, anom_samples])
    return samples

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
        - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        - https://pytorch.org/vision/main/auto_examples/transforms/plot_custom_transforms.html

    """
    def __init__(self, mean_list, alpha=50) -> None:
        self.alpha = alpha
        self.mean_list = mean_list


    def __call__(self, x, idx):
        x = x.to('cpu').detach().numpy().copy()

        # You Transform with numpy
        I_mean = self.get_image_mean(x) if self.mean_list[idx] is None else self.mean_list[idx]
        x = np.arctan((x - I_mean) / I_mean) * self.alpha + I_mean 

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



        assert len(vol_idxs)==len(img_paths),\
            f"{len(mean_list)=}\t{len(img_paths)=}"

        mean_list = list()
        prev_idx = None
        for idx in vol_idxs:
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


if __name__ == "__main__":

    logging.basicConfig(filename='./logs/debug.log', filemode='w', level=logging.DEBUG)

#     img_list = pd.read_csv(img_set_path)["img_path"].unique()
#     mean_list = YouTransform.get_volume_mean(img_list)
#     transform = myCompose([
#                     transforms.Resize(
#                         (256, 256),
#                         interpolation=transforms.InterpolationMode.BILINEAR
#                     ),
#                     transforms.ToTensor(), # [0,255]->[0,1]
#                     YouTransform(examples["mean"], alpha=0.2, ), # 50/255
#                 ])

    # https://pytorch.org/vision/main/transforms.html#v2-api-ref
    transforms = v2.Compose([
                    v2.Grayscale(),
                    v2.PILToTensor(),
                    v2.Resize(
                        size=(256, 256), 
                        interpolation=v2.InterpolationMode.BILINEAR
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    # v2.Normalize(mean=[0.485], std=[0.229]),
                    # YouTransform(examples["mean"], alpha=0.2, )
                ]) 

    # root = "/home/skazuki/data/WDDD2_AD" 
    root = "/mnt/e/WDDD2_AD"

    length=10
    for phase in ("train", "val", "test"):
        for l in range(length):
            img_path = f"{root}/wildType/{phase}/anomaly/{l:0=3}.png"
            msk_path = f"{root}/wildType/ground_truth/anomaly/{l:0=3}_mask.png"
            gen_anomalous(img_path, msk_path, size=(600,600,),)
     

    datamodule = WDDD2_AD(
        root = root,  
        category = "wildType",
        train_batch_size = 32,
        eval_batch_size = 32,
        num_workers = 30,
        task = TaskType.SEGMENTATION,# CLASSIFICATION,#  
        val_split_mode = ValSplitMode.NONE,# .FROM_TEST,
        # val_split_ratio = 0.01,
        image_size = (256,256),
        transform = transforms,
        seed  = 44,
        debug =True, 
        debug_data_ratio =0.01, 
        add_anomalous = True, # False, 
    )
    print("prepareing datamodule...")
    datamodule.setup()


    # "train_data", "val_data", "test_data"
    for attribute in ("train_data", "val_data", "test_data"):
        data_loader = getattr(datamodule,attribute)
        print(f"total iterations: {len(data_loader)}")
        i, batch = next(enumerate(data_loader))

        print(batch.keys()) # dict_keys(['image_path', 'label', 'image', 'mask'])
        print(batch["image_path"])
        print(batch["label"])
        print(batch["image"].shape)
        print(batch["image"].shape)
        print()


