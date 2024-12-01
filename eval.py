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
from src.util.logging_util import tb_logger, eval_dic_to_text

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
    parser.add_argument(
        "--do_not_copy_data",
        action="store_true",
        help="On Slurm cluster, do not copy data to local scratch",
    )
    # parser.add_argument(
    #     "--base_data_dir", type=str, default='./data/hypersim/dataset_pp', help="directory of training data"
    # )
    parser.add_argument(
        "--base_data_dir", type=str, default='./data/sam/pix2gestalt_occlusions_release', help="directory of training data"
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_true",
        help="Add datetime to the output folder name",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--trained_checkpoint",
        type=str,
        default="./",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
    )
    
    args = parser.parse_args()
    resume_run = args.resume_run
    output_dir = args.output_dir
    base_data_dir = (
        args.base_data_dir
        if args.base_data_dir is not None
        else os.environ["BASE_DATA_DIR"]
    )

    # -------------------- Initialization --------------------
    # Resume previous run
    if resume_run is not None:
        raise NotImplementedError("Resume running is not supported yet.")
        
    # Run from start
    cfg = recursive_load_config(args.config)
    
    # -------------------- Gradient accumulation steps --------------------
    eff_bs = cfg.dataloader.effective_batch_size
    accumulation_steps = eff_bs / (cfg.dataloader.max_train_batch_size * torch.cuda.device_count())
    assert int(accumulation_steps) == accumulation_steps
    accumulation_steps = int(accumulation_steps)

    # ddp
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps = accumulation_steps, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # Full job name
    timestamp = torch.tensor(time.time(), dtype=torch.float64).to(device)
    accelerate.utils.broadcast(timestamp)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp.item()))

    # Output dir
    if output_dir is None:
        out_dir_run = os.path.join("./work_dir", "default_folder")
    else:
        out_dir_run = output_dir
        
    # os.makedirs(out_dir_run, exist_ok=False)
    if accelerator.is_main_process:
        os.makedirs(out_dir_run, exist_ok=True) # when debugging, overwrite the existing folder
        out_dir_run = os.path.join(out_dir_run, timestamp)
        os.makedirs(out_dir_run, exist_ok=True)
        
    # Other directories
    out_dir_eval = os.path.join(out_dir_run, "evaluation")
    out_dir_vis = os.path.join(out_dir_run, "visualization")
    
    if accelerator.is_main_process:
        if not os.path.exists(out_dir_eval):
            os.makedirs(out_dir_eval)
        if not os.path.exists(out_dir_vis):
            os.makedirs(out_dir_vis)
        
    # -------------------- Logging settings --------------------
    accelerator.wait_for_everyone()
    config_logging(cfg.logging, out_dir=out_dir_run)
    if accelerator.is_main_process:
        logging.debug(f"config: {cfg}")

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    # device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"device = {device}")

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
    
    if accelerator.is_main_process:
        logging.debug("Augmentation: ", cfg.augmentation)
    if "mixed" == cfg_data.train.name:
        dataset_ls = train_dataset
        assert len(cfg_data.train.prob_ls) == len(
            dataset_ls
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=cfg.dataloader.max_train_batch_size,
            drop_last=True,
            prob=cfg_data.train.prob_ls,
            shuffle=True,
            generator=loader_generator,
        )
        train_loader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            num_workers=cfg.dataloader.num_workers,
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.dataloader.max_train_batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=True,
            generator=loader_generator,
        )
    # Validation dataset
    val_loaders: List[DataLoader] = []
    for _val_dic in cfg_data.val:
        _val_dataset = get_dataset(
            _val_dic,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
            depth_transform=depth_transform,
        )
        _val_loader = DataLoader(
            dataset=_val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
        )
        _val_loader = accelerator.prepare(_val_loader)
        val_loaders.append(_val_loader)

    # Visualization dataset
    vis_loaders: List[DataLoader] = []
    for _vis_dic in cfg_data.vis:
        _vis_dataset = get_dataset(
            _vis_dic,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
            depth_transform=depth_transform,
        )
        _vis_loader = DataLoader(
            dataset=_vis_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
        )
        vis_loaders.append(_vis_loader)

    
    # -------------------- Model --------------------
    model = get_model(cfg.model.name, **cfg.model.kwargs) # delay loading pre-trained model
    model = model.from_pretrained(args.trained_checkpoint, strict=True)

    # -------------------- Trainer --------------------
    # Exit time
    if args.exit_after > 0:
        t_end = t_start + timedelta(minutes=args.exit_after)
        if accelerator.is_main_process:
            logging.info(f"Will exit at {t_end}")
    else:
        t_end = None

    trainer_cls = get_trainer_cls(cfg.trainer.name)
    if accelerator.is_main_process:
        logging.debug(f"Trainer: {trainer_cls}")
    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=train_loader,
        device=device,
        out_dir_ckpt=None,
        out_dir_eval=out_dir_eval,
        out_dir_vis=out_dir_vis,
        accumulation_steps=accumulation_steps,
        val_dataloaders=val_loaders,
        vis_dataloaders=vis_loaders,
        accelerator=accelerator,
    )

    # -------------------- Checkpoint --------------------
    if resume_run is not None:
        trainer.load_checkpoint(
            resume_run, load_trainer_state=True, resume_lr_scheduler=True
        )

    # -------------------- Training & Evaluation Loop --------------------
    for i, val_loader in enumerate(trainer.val_loaders):
        val_dataset_name = val_loader.dataset.disp_name
        val_dict = trainer.validate_single_dataset(data_loader=val_loader, eval=True)
        for metric_k, metric_v in val_dict.items():
            text = eval_dic_to_text(
                val_metrics=metric_v,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
                diff=metric_k,
            )
            if accelerator.is_main_process:
                print(text)
                with open(os.path.join(out_dir_eval, "eval.txt"), "a") as f:
                    f.write(text)
                    f.write("\n")