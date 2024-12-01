import argparse
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import List

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from src.models import get_model

# from marigold import get_pipeline
from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.mixed_sampler import MixedBatchSampler
from src.trainer import get_trainer_cls
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)
from src.util.depth_transform import (
    DepthNormalizerBase,
    get_depth_normalizer,
)
from src.util.logging_util import (
    config_logging,
    init_wandb,
    load_wandb_job_id,
    log_slurm_job_id,
    save_wandb_job_id,
    tb_logger,
)
from src.util.slurm_util import get_local_scratch_dir, is_on_slurm

# ddp support
import time
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

if "__main__" == __name__:
    t_start = datetime.now()

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_marigold.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config, and checkpoint in the config",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="directory to save checkpoints"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Do not use cuda.")
    parser.add_argument(
        "--exit_after",
        type=int,
        default=-1,
        help="Save checkpoint and exit after X minutes.",
    )
    parser.add_argument("--no_wandb", action="store_true", help="run without wandb")
    # parser.add_argument(
    #     "--base_data_dir", type=str, default='./data/hypersim/dataset_pp', help="directory of training data"
    # )
    parser.add_argument(
        "--base_data_dir", type=str, default='./data/sam/pix2gestalt_occlusions_release', help="directory of training data"
    )

    args = parser.parse_args()
    resume_run = args.resume_run
    output_dir = args.output_dir
    base_data_dir = (
        args.base_data_dir
        if args.base_data_dir is not None
        else os.environ["BASE_DATA_DIR"]
    )

    cfg = recursive_load_config(args.config)
    
    # -------------------- Data --------------------
    cfg_data = cfg.dataset
    loader_seed = cfg.dataloader.seed
    if loader_seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(loader_seed)

    # Training dataset
    depth_transform: DepthNormalizerBase = get_depth_normalizer(
        cfg_normalizer=cfg.depth_normalization
    )
    train_dataset: BaseDepthDataset = get_dataset(
        cfg_data.train,
        base_data_dir=base_data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        depth_transform=depth_transform,
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=cfg.dataloader.num_workers,
        shuffle=True,
        generator=loader_generator,
    )
    
    # Validation dataset
    val_dataset = get_dataset(
        cfg_data.val[0],
        base_data_dir=base_data_dir,
        mode=DatasetMode.EVAL,
        depth_transform=depth_transform,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
    )

    cls_dict = {
        'small_obj_vis_ratio_small': 0,
        'small_obj_vis_ratio_medium': 0,
        'small_obj_vis_ratio_large': 0,
        'medium_obj_vis_ratio_small': 0,
        'medium_obj_vis_ratio_medium': 0,
        'medium_obj_vis_ratio_large': 0,
        'large_obj_vis_ratio_small': 0,
        'large_obj_vis_ratio_medium': 0,
        'large_obj_vis_ratio_large': 0,
    }
    
    cls_v2_dict = {
        'ratio_small': 0,
        'ratio_medium': 0,
        'ratio_large': 0,
    }

    idx = 0
    # for batch in tqdm(train_loader):
    for batch in tqdm(val_loader):

        idx += 1
        
        object_mask = batch["guide"]
        visibility_mask = batch["visible_mask"]
        
        image_size_h, image_size_w = object_mask.shape[-2], object_mask.shape[-1]
        
        object_mask_pixel_num = torch.sum(object_mask > 0)
        visibility_mask_pixel_num = torch.sum(visibility_mask > 0)
        object_ratio = object_mask_pixel_num / (image_size_h * image_size_w)
        visibility_ratio = visibility_mask_pixel_num / object_mask_pixel_num
        
        if object_ratio > 0.25:
            object_size = 'large'
        elif object_ratio > 0.0625:
            object_size = 'medium'
        else:
            object_size = 'small'
            
        # if visibility_ratio > 0.8:
        #     visibility_size = 'large'
        # elif visibility_ratio > 0.5:
        #     visibility_size = 'medium'
        # else:
        #     visibility_size = 'small'
        
        if visibility_ratio > 0.75:
            visibility_size = 'large'
        elif visibility_ratio > 0.5:
            visibility_size = 'medium'
        else:
            visibility_size = 'small'
        
        cls_dict[f'{object_size}_obj_vis_ratio_{visibility_size}'] += 1
        cls_v2_dict[f'ratio_{visibility_size}'] += 1

    # print(cls_dict['small_obj_vis_ratio_small'], cls_dict['small_obj_vis_ratio_medium'], cls_dict['small_obj_vis_ratio_large'])
    # print(cls_dict['medium_obj_vis_ratio_small'], cls_dict['medium_obj_vis_ratio_medium'], cls_dict['medium_obj_vis_ratio_large'])
    # print(cls_dict['large_obj_vis_ratio_small'], cls_dict['large_obj_vis_ratio_medium'], cls_dict['large_obj_vis_ratio_large'])
    
    # print(cls_dict['small_obj_vis_ratio_small'] + cls_dict['medium_obj_vis_ratio_small'] + cls_dict['small_obj_vis_ratio_medium']) # hard
    # print(cls_dict['large_obj_vis_ratio_small'] + cls_dict['medium_obj_vis_ratio_medium'] + cls_dict['small_obj_vis_ratio_large']) # mid
    # print(cls_dict['large_obj_vis_ratio_medium'] + cls_dict['medium_obj_vis_ratio_large'] + cls_dict['large_obj_vis_ratio_large']) # easy

    # validation set info:
    # 298 423 174
    # 484 958 413
    # 245 662 343
    # 1205
    # 1377
    # 1418
    
    print(cls_v2_dict)