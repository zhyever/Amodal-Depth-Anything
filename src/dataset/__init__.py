# This file is modified based on the original file from the Marigold repository (xx)

import os

from .base_depth_dataset import BaseDepthDataset, get_pred_name, DatasetMode  # noqa: F401
from .sam_amodal_dataset import SAMAmodalDataset

dataset_name_class_dict = {}

def get_dataset(
    cfg_data_split, base_data_dir: str, mode: DatasetMode, **kwargs
) -> BaseDepthDataset:
    if "mixed" == cfg_data_split.name:
        assert DatasetMode.TRAIN == mode, "Only training mode supports mixed datasets."
        dataset_ls = [
            get_dataset(_cfg, base_data_dir, mode, **kwargs)
            for _cfg in cfg_data_split.dataset_list
        ]
        return dataset_ls
    elif cfg_data_split.name == 'sam': # for sam dataset: the folder structure is different from the other datasets
        dataset_class = SAMAmodalDataset
        dataset = dataset_class(
            mode=mode,
            filename_ls_path=cfg_data_split.filenames,
            dataset_dir=base_data_dir, # no need to join with cfg_data_split.dir
            **cfg_data_split,
            **kwargs,
        )
    elif cfg_data_split.name in dataset_name_class_dict.keys():
        dataset_class = dataset_name_class_dict[cfg_data_split.name]
        dataset = dataset_class(
            mode=mode,
            filename_ls_path=cfg_data_split.filenames,
            dataset_dir=os.path.join(base_data_dir, cfg_data_split.dir),
            **cfg_data_split,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset
