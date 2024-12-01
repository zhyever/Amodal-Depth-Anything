
import logging
import os
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
from diffusers import DDPMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# from marigold.marigold_pipeline import MarigoldPipeline, MarigoldDepthOutput
# from marigold.marigold_amodal_pipeline import MarigoldAmodalPipeline
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dic_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.alignment import align_depth_least_square
from src.util.seeding import generate_seed_sequence
from src.util.image_util import chw2hwc, colorize_depth_maps
import torch.nn.functional as F
from src.models.depthfm.unet.util import conv_nd

class DepthFMAmodalTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model,
        train_dataloader: DataLoader,
        device,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
        accelerator = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps

        # Trainability
        self.model.vae.requires_grad_(False) # vae
        # self.model.empty_text_embed.requires_grad_(False) would be a np.array
        self.model.model.requires_grad_(True) # unet

        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr * self.cfg.scale_lr
        self.optimizer = Adam(self.model.model.parameters(), lr=lr)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter * accelerator.num_processes,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps * accelerator.num_processes,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(*["loss"])
        self.val_easy_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        self.val_mid_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        self.val_diff_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        self.val_align_easy_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        self.val_align_mid_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        self.val_align_diff_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        self.val_align_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        self.metric_dict = {
            "easy": self.val_easy_metrics,
            "mid": self.val_mid_metrics,
            "diff": self.val_diff_metrics,
            "overall": self.val_metrics,
            "align_easy": self.val_align_easy_metrics,
            "align_mid": self.val_align_mid_metrics,
            "align_diff": self.val_align_diff_metrics,
            "align_overall": self.val_align_metrics,}
        
        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.gt_depth_type = self.cfg.gt_depth_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

        # ddp
        self.accelerator = accelerator
        self.model, self.optimizer, self.train_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.lr_scheduler)
        
        self.max_grad_norm = self.cfg.trainer.max_grad_norm
        
    def train(self, t_end=None):
        if self.accelerator.is_main_process:
            logging.info("Start training")

        device = self.device
        self.model.to(device)
        
        if self.in_evaluation:
            if self.accelerator.is_main_process:
                logging.info(
                    "Last evaluation was not finished, will do evaluation before continue training."
                )
                self.validate()

        self.train_metrics.reset()

        if self.accelerator.is_main_process:
            pbar = tqdm(total = self.max_iter)
        
        accumulated_step = 0
        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            if self.accelerator.is_main_process:
                logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                with self.accelerator.accumulate(self.model):
                    self.model.train()

                    # globally consistent random generators
                    if self.seed is not None:
                        local_seed = self._get_next_seed()
                        rand_num_generator = torch.Generator(device=device)
                        rand_num_generator.manual_seed(local_seed)
                    else:
                        rand_num_generator = None

                    # Get data
                    rgb = batch["rgb_norm"].to(device)
                    depth_gt_for_latent = batch[self.gt_depth_type].to(device)
                    
                    if self.gt_mask_type is not None:
                        valid_mask_for_latent = batch[self.gt_mask_type].to(device)
                        # valid_mask_down = valid_mask_for_latent.repeat((1, 4, 1, 1))
                        invalid_mask = ~valid_mask_for_latent
                        valid_mask_down = ~torch.max_pool2d(
                            invalid_mask.float(), 8, 8).bool() # will be extended a little bit
                        valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))
                    else:
                        raise NotImplementedError

                    model_pred, target = self.model(
                        ims=rgb, 
                        num_steps=4, 
                        ensemble_size=1, 
                        mode='train', 
                        depth=depth_gt_for_latent, 
                        rand_num_generator=rand_num_generator,
                        guide_mask=batch['guide'].to(device),  # 0-1
                        guide_rgb=batch['guide_rgb_norm'].to(device), # -1-1
                        observation=batch['depth_observation'].to(device)) # 0-1

                    # model_pred = F.interpolate(model_pred, size=(valid_mask_for_latent.shape[2], valid_mask_for_latent.shape[3]), mode='bilinear', align_corners=True)
                    # target = F.interpolate(target, size=(valid_mask_for_latent.shape[2], valid_mask_for_latent.shape[3]), mode='bilinear', align_corners=True)
                    
                    if torch.isnan(model_pred).any():
                        logging.warning("model_pred contains NaN.")

                    # Get the target for loss depending on the prediction type
                    if self.gt_mask_type is not None:
                        entire_mask = batch['guide'].to(device)
                        entire_mask = F.interpolate(entire_mask.float(), size=(model_pred.shape[2], model_pred.shape[3])) > 0
                            
                        invisible_mask = batch['invisible_mask'].to(device)
                        invisible_mask = F.interpolate(invisible_mask.float(), size=(model_pred.shape[2], model_pred.shape[3])) > 0
                        
                        visible_mask = batch['visible_mask'].to(device)
                        visible_mask = F.interpolate(visible_mask.float(), size=(model_pred.shape[2], model_pred.shape[3])) > 0
                        
                        if self.cfg.trainer.loss_stategy == 'invisible_part':
                            invisible_mask = torch.logical_and(valid_mask_down, invisible_mask)
                            latent_loss = self.loss(
                                model_pred[invisible_mask].float(),
                                target[invisible_mask].float())
                            
                        elif self.cfg.trainer.loss_stategy == 'entire_target_object':
                            entire_mask = torch.logical_and(valid_mask_down, entire_mask)
                            latent_loss = self.loss(
                                model_pred[entire_mask].float(),
                                target[entire_mask].float())
                        
                        elif self.cfg.trainer.loss_stategy == 'entire_scene':
                            scene_mask = valid_mask_down
                            latent_loss = self.loss(
                                model_pred[scene_mask].float(),
                                target[scene_mask].float())
                        
                    else:
                        raise NotImplementedError
                        latent_loss = self.loss(model_pred.float(), target.float())

                    loss = latent_loss.mean()

                    self.train_metrics.update("loss", loss.item())

                    self.accelerator.backward(loss)
                    self.n_batch_in_epoch += 1

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, norm_type = 2)
        
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    accumulated_step += 1
                    if accumulated_step >= self.gradient_accumulation_steps:
                        self.effective_iter += 1
                        accumulated_step = 0
                        
                        # Log to tensorboard
                        log_loss = self.train_metrics.result()["loss"]
                    
                        if self.accelerator.is_main_process:
                            if self.effective_iter % self.cfg.logging.log_interval == 0:
                                tb_logger.log_dic(
                                    {
                                        f"train/{k}": v
                                        for k, v in self.train_metrics.result().items()
                                    },
                                    global_step=self.effective_iter,
                                )
                                tb_logger.writer.add_scalar(
                                    "lr",
                                    self.lr_scheduler.get_last_lr()[0],
                                    global_step=self.effective_iter,
                                )
                                tb_logger.writer.add_scalar(
                                    "n_batch_in_epoch",
                                    self.n_batch_in_epoch,
                                    global_step=self.effective_iter,
                                )
                                logging.info(
                                    f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={log_loss:.5f}"
                                )
                            
                            pbar.update(1)
                            pbar.set_description('Current loss: {:.2f}'.format(log_loss))
                        
                        self.train_metrics.reset()

                        # Per-step callback
                        self._train_step_callback()
                        
                        # End of training
                        if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                            if self.accelerator.is_main_process:
                                self.save_checkpoint(
                                    ckpt_name=self._get_backup_ckpt_name(),
                                    save_train_state=False,
                                )
                                logging.info("Training ended.")
                            return
                        # Time's up
                        elif t_end is not None and datetime.now() >= t_end:
                            if self.accelerator.is_main_process:
                                self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                                logging.info("Time is up, training paused.")
                            return

                        torch.cuda.empty_cache()
                        # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0
        
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def encode_depth(self, depth_in):
        # stack depth into 3-channel
        stacked = self.stack_depth_images(depth_in)
        # encode using VAE encoder
        depth_latent = self.model.encode_rgb(stacked)
        return depth_latent

    @staticmethod
    def stack_depth_images(depth_in):
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1)
            stacked = depth_in.repeat(1, 3, 1, 1)
        return stacked

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            if self.accelerator.is_main_process:
                self.save_checkpoint(
                    ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                )

        _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            # self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            # if self.accelerator.is_main_process:
            #     self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            # _is_latest_saved = True
            self.validate()
            # self.in_evaluation = False
            # if self.accelerator.is_main_process:
            #     self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            if self.accelerator.is_main_process:
                self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            if self.accelerator.is_main_process:
                self.visualize()
        
        self.accelerator.wait_for_everyone()

    def validate(self):
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            val_dict = self.validate_single_dataset(data_loader=val_loader, eval=True)
            if self.accelerator.is_main_process:
                logging.info(
                    f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`")
                
                _save_to = os.path.join(
                    self.out_dir_eval,
                    f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
                )
                
                for metric_k, metric_v in val_dict.items():
                    tb_logger.log_dic(
                        {f"{metric_k}/{val_dataset_name}/{k}": v for k, v in metric_v.items()},
                        global_step=self.effective_iter,
                    )
                    
                    text = eval_dic_to_text(
                        val_metrics=metric_v,
                        dataset_name=val_dataset_name,
                        sample_list_path=val_loader.dataset.filename_ls_path,
                        diff=metric_k,
                    )
                
                    with open(_save_to, "a") as f:
                        f.write(text)
                        f.write("\n")
                        
                # Update main eval metric
                if 0 == i:
                    main_eval_metric = val_dict['align_overall'][self.main_val_metric]
                    if (
                        "minimize" == self.main_val_metric_goal
                        and main_eval_metric < self.best_metric
                        or "maximize" == self.main_val_metric_goal
                        and main_eval_metric > self.best_metric
                    ):
                        self.best_metric = main_eval_metric
                        logging.info(
                            f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                        )
                        # Save a checkpoint
                        # self.save_checkpoint(
                        #     ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                        # )

    def visualize(self):
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                save_to_dir=vis_out_dir,
            )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        save_to_dir: str = None,
        eval = False,
    ):
        self.model.eval()
        self.model.to(self.device)
        self.val_metrics.reset()
        self.val_easy_metrics.reset()
        self.val_mid_metrics.reset() 
        self.val_diff_metrics.reset()
        self.val_align_metrics.reset()
        self.val_align_easy_metrics.reset()
        self.val_align_mid_metrics.reset() 
        self.val_align_diff_metrics.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

        pbar = tqdm(enumerate(data_loader, start=1), desc=f"evaluating on {data_loader.dataset.disp_name}", total=len(data_loader)) if self.accelerator.is_main_process \
            else enumerate(data_loader, start=1)
        
        for i, batch in pbar:
            # Read input image
            rgb = batch["rgb_norm"].to(self.device)
            
            # GT depth
            depth_raw_ts = batch["depth_gt"].squeeze()
            depth_raw = depth_raw_ts.detach().cpu().numpy()
            depth_raw_ts = depth_raw_ts.to(self.device)

            valid_mask_ts = batch["valid_mask_raw"].squeeze()
            valid_mask = valid_mask_ts.detach().cpu().numpy()
            valid_mask_ts = valid_mask_ts.to(self.device)
            
            guide_mask_ts = batch['guide'].squeeze()
            guide_mask_ts = guide_mask_ts.to(self.device)
            guide_mask_ts = torch.logical_and(guide_mask_ts, valid_mask_ts)
            
            visible_mask_ts = batch["visible_mask"].squeeze()
            visible_mask = visible_mask_ts.detach().cpu().numpy()
            visible_mask_ts = visible_mask_ts.to(self.device)
            
            depth_observation_ts = batch["depth_observation"].squeeze()
            depth_observation = depth_observation_ts.detach().cpu().numpy()
            depth_observation_ts = depth_observation_ts.to(self.device)
            
            object_ts = batch["invisible_mask"].squeeze()
            object_ts = object_ts.to(self.device)
            object_ts = torch.logical_and(object_ts, valid_mask_ts)
            
            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            depth_pred_ts = self.model(
                        ims=rgb, 
                        num_steps=4, 
                        ensemble_size=1, 
                        mode='infer', 
                        depth=None, 
                        rand_num_generator=generator,
                        guide_mask=batch['guide'].to(self.device),  # 0-1
                        guide_rgb=batch['guide_rgb_norm'].to(self.device), # -1-1
                        observation=batch['depth_observation'].to(self.device)) # 0-1
            pred = depth_pred_ts.squeeze()
            pred_np = pred.detach().cpu().numpy()
            depth_pred_np = depth_pred_ts.detach().cpu().numpy()

            # align depth
            # depth_align, scale, shift = align_depth_least_square(
            #     gt_arr=depth_raw,
            #     pred_arr=depth_pred_np,
            #     valid_mask_arr=guide_mask_ts.detach().cpu(),
            #     return_scale_shift=True,
            #     max_resolution=None)
            depth_align, scale, shift = align_depth_least_square(
                gt_arr=depth_observation, # utilize the observation!
                pred_arr=pred_np,
                valid_mask_arr=visible_mask_ts.detach().cpu().bool(), # only use the visible part to do this alignment
                return_scale_shift=True,
                max_resolution=None)
            depth_align = torch.tensor(depth_align).cuda()
            
            # check difficulty
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
                
            if visibility_ratio > 0.75:
                visibility_size = 'large'
            elif visibility_ratio > 0.5:
                visibility_size = 'medium'
            else:
                visibility_size = 'small'
            
            if visibility_size == 'small':
                select_tracker = self.val_diff_metrics
                select_tracker_align = self.val_align_diff_metrics
            elif visibility_size == 'medium':
                select_tracker = self.val_mid_metrics
                select_tracker_align = self.val_align_mid_metrics
            elif visibility_size == 'large':
                select_tracker = self.val_easy_metrics
                select_tracker_align = self.val_align_easy_metrics
            else:
                raise NotImplementedError
    
            # Evaluate
            sample_metric = []
            if eval:
                for met_func in self.metric_funcs:
                    _metric_name = met_func.__name__
                    # only evaluate the invisible part
                    _metric = met_func(pred + 1e-5, depth_raw_ts + 1e-5, object_ts)
                    # _metric = met_func(pred + 1e-5, depth_raw_ts + 1e-5, guide_mask_ts)
                    
                    if self.accelerator.state.num_processes > 1:
                        gather_metric = self.accelerator.gather_for_metrics(_metric.to(self.device))
                    else:
                        gather_metric = [_metric.to(self.device)]
                    sample_metric.append(_metric.__str__())
                    for m in gather_metric:
                        if torch.isnan(m).any():
                            continue # skip nan case
                        self.val_metrics.update(_metric_name, m.item())
                        select_tracker.update(_metric_name, m.item())
                        
                    # align version
                    # _metric = met_func(depth_align + 1e-5, depth_raw_ts + 1e-5, guide_mask_ts)
                    _metric = met_func(depth_align + 1e-5, depth_raw_ts + 1e-5, object_ts)
                    
                    if self.accelerator.state.num_processes > 1:
                        gather_metric = self.accelerator.gather_for_metrics(_metric.to(self.device))
                    else:
                        gather_metric = [_metric.to(self.device)]
                    sample_metric.append(_metric.__str__())
                    for m in gather_metric:
                        if torch.isnan(m).any():
                            continue # skip nan case
                        self.val_align_metrics.update(_metric_name, m.item())
                        select_tracker_align.update(_metric_name, m.item())

            # this save_to_dir is for validation during training.
            if save_to_dir is not None:
                if self.accelerator.is_main_process:
                    img_name = batch["rgb_relative_path"][0].replace("/", "_")
                    png_save_path = os.path.join(save_to_dir, f"{img_name}_{self.accelerator.process_index}.png")
                    
                    if 'ssi' in self.cfg.trainer.loss_stategy:
                        depth_colored = colorize_depth_maps(depth_align.detach().cpu().numpy(), 0, 1, cmap='Spectral').squeeze()
                        depth_colored[torch.logical_not(torch.stack([batch['guide'].squeeze(), batch['guide'].squeeze(), batch['guide'].squeeze()], dim=0))] = 0
                    else:
                        depth_colored = colorize_depth_maps(pred_np, 0, 1, cmap='Spectral').squeeze()  # [3, H, W], value in (0, 1)
                        depth_colored[torch.logical_not(torch.stack([batch['guide'].squeeze(), batch['guide'].squeeze(), batch['guide'].squeeze()], dim=0))] = 0
                        
                    depth_colored = (depth_colored * 255).astype(np.uint8)
                    depth_colored_hwc = chw2hwc(depth_colored)
                    # depth_colored_img = Image.fromarray(depth_colored_hwc)
                    # depth_colored_img.save(png_save_path)
                    
                    depth_gt_colored = colorize_depth_maps(depth_raw, 0, 1, cmap='Spectral').squeeze()  # [3, H, W], value in (0, 1)
                    depth_gt_colored = (depth_gt_colored * 255).astype(np.uint8)
                    depth_gt_colored_hwc = chw2hwc(depth_gt_colored)
                    # depth_gt_colored_img = Image.fromarray(depth_gt_colored_hwc)
                    # depth_gt_colored_img.save(png_save_path)
                    
                    rgb_numpy = batch["rgb_int"].squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                    # rgb_numpy = Image.fromarray(rgb_numpy)
                    # rgb_numpy.save(png_save_path.replace('.png', '_rgb.png'))
                    
                    rgb_raw_numpy = batch["rgb_int"].squeeze().permute(1, 2, 0).detach().cpu()
                    guide = batch['guide'].squeeze().unsqueeze(dim=-1).detach().cpu().repeat(1, 1, 3).numpy() > 0
                    rgb_raw_numpy[guide] = 0
                    
                    # .numpy().astype(np.uint8)
                    # rgb_numpy = Image.fromarray(rgb_numpy)
                    # rgb_numpy.save(png_save_path.replace('.png', '_rgb.png'))
                    
                    save_temp = np.zeros((depth_colored_hwc.shape[0] * 2, depth_colored_hwc.shape[1] * 2, 3), dtype=np.uint8)
                    save_temp[:depth_colored_hwc.shape[0], :depth_colored_hwc.shape[1], :] = depth_colored_hwc
                    save_temp[depth_colored_hwc.shape[0]:, :depth_colored_hwc.shape[1], :] = depth_gt_colored_hwc
                    save_temp[:depth_colored_hwc.shape[0], depth_colored_hwc.shape[1]:, :] = rgb_numpy
                    save_temp[depth_colored_hwc.shape[0]:, depth_colored_hwc.shape[1]:, :] = rgb_raw_numpy
                    save_temp = Image.fromarray(save_temp)
                    save_temp.save(png_save_path)
    
        
        self.model.train()
        
        return_dict = {
            "easy": self.val_easy_metrics.result(),
            "mid": self.val_mid_metrics.result(),
            "diff": self.val_diff_metrics.result(),
            "overall": self.val_metrics.result(),
            "align_easy": self.val_align_easy_metrics.result(),
            "align_mid": self.val_align_mid_metrics.result(),
            "align_diff": self.val_align_diff_metrics.result(),
            "align_overall": self.val_align_metrics.result(),}
        
        return return_dict
    
    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            if self.accelerator.is_main_process:
                logging.info(
                    f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
                )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        # unet_path = os.path.join(ckpt_dir, "unet")
        unet_path = ckpt_dir
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            unet_path, 
            safe_serialization=False, 
            is_main_process=self.accelerator.is_main_process, 
            save_function=self.accelerator.save)
        logging.info(f"UNet is saved to: {unet_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
        self.model.unet.load_state_dict(
            torch.load(_model_path, map_location=self.device)
        )
        self.model.unet.to(self.device)
        logging.info(f"UNet parameters are loaded from {_model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"
