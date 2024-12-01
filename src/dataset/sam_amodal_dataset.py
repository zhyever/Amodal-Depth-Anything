from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode, DatasetMode
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode, Resize
import random

class SAMAmodalDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # Hypersim data parameter
            min_depth=0,
            max_depth=1,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.rgb_i_d,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode SAM depth
        depth_decoded = depth_in / 65535
        return depth_decoded

    def _infer_preprocess(self, rasters):
        return rasters
    
    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)
        
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
        else:
            # Resize
            if self.resize_to_hw is not None:
                resize_transform = Resize(
                    size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT)
                rasters = {k: resize_transform(v) for k, v in rasters.items()}

        invisible_mask = torch.logical_and(torch.logical_not(rasters['visible_mask']), rasters['guide'])
        rasters['invisible_mask'] = invisible_mask
        
        # merge
        outputs = rasters
        outputs.update(other)
        
        return outputs
    
    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        rgb_rel_path = filename_line[0]
        depth_rel_path = filename_line[1]
        rgb_rel_path = depth_rel_path.replace("depth", "occlusion")
        visible_path = depth_rel_path.replace("depth", "visible_object_mask")
        visible_path = visible_path.replace("_visible_object_mask.png", "_visible_mask.png")
        guide_path = depth_rel_path.replace("depth", "whole_mask")
        
        depth_rel_path, filled_rel_path = None, None
        if DatasetMode.RGB_ONLY != self.mode:
            # observation, gt
            depth_rel_path = [filename_line[1].replace("depth/", "depth_da_update_occ/"), filename_line[1].replace("depth/", "depth_da_update_combine/")] # HACK: to switch to another version of depth pesudo labels
            if self.has_filled_depth:
                filled_rel_path = filename_line[2]
        return rgb_rel_path, depth_rel_path, filled_rel_path, visible_path, guide_path
    
    
    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        
        outputs = {
            "rgb_int": torch.from_numpy(rgb).int(),
            "rgb_norm": torch.from_numpy(rgb_norm).float()}
        
        return outputs
    
    def _load_depth_data(self, depth_rel_path, filled_rel_path=None):
        # Read depth data
        outputs = {}
        depth_observation = self._read_depth_file(depth_rel_path[0]).squeeze()
        depth_observation = torch.from_numpy(depth_observation).float().unsqueeze(0)  # [1, H, W]
        outputs["depth_observation"] = depth_observation.clone()
        depth_gt = self._read_depth_file(depth_rel_path[1]).squeeze()
        depth_gt = torch.from_numpy(depth_gt).float().unsqueeze(0)  # [1, H, W]
        outputs["depth_gt"] = depth_gt.clone()
        return outputs
    
    def _training_preprocess(self, rasters):
        # Augmentation
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)

        # remove depth normalization
        # remove setting invalid pixel to far plane
        # imagination drop out:
        if self.img_dropout > 0.0:
            if random.random() < self.img_dropout:
                # if random.random() < 0.5:
                #     # rasters["guide_rgb_int"] = torch.ones_like(rasters["guide_rgb_int"]) * (-2.0)
                #     # rasters["guide_rgb_norm"] = torch.ones_like(rasters["guide_rgb_norm"]) * (-2.0)
                #     rasters["guide_rgb_int"] = torch.ones_like(rasters["guide_rgb_int"]) * 0.0
                #     rasters["guide_rgb_norm"] = torch.ones_like(rasters["guide_rgb_norm"]) * 0.0
                # else:
                #     # rasters["guide"] = torch.ones_like(rasters["guide"]) * (-0.5)
                #     # rasters["guide"] = torch.ones_like(rasters["guide"]) * 0.0
                #     rasters["guide"] = torch.ones_like(rasters["guide"]) * 0.5
                rasters["guide_rgb_int"] = torch.ones_like(rasters["guide_rgb_int"]) * 0.0
                rasters["guide_rgb_norm"] = torch.ones_like(rasters["guide_rgb_norm"]) * 0.0
        
        # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
            )
            rasters = {k: resize_transform(v) for k, v in rasters.items()}

        return rasters
    
    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, filled_rel_path, visible_path, guide_path = self._get_data_path(index=index)
        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))
        rgb_guide = self._load_rgb_data(rgb_rel_path=rgb_rel_path.replace("occlusion", "whole"))
        rasters.update({'guide_rgb_int':rgb_guide['rgb_int'].float()})
        rasters.update({'guide_rgb_norm':rgb_guide['rgb_norm'].float()})
        guide_mask = torch.from_numpy(self._read_image(img_rel_path=guide_path) > 0).unsqueeze(0).float()
        visible_mask = torch.from_numpy(self._read_image(img_rel_path=visible_path) > 0).unsqueeze(0).float()
        rasters.update({'guide': guide_mask, 'visible_mask': visible_mask})
        
        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, filled_rel_path=filled_rel_path)
            rasters.update(depth_data)
            # valid mask: all pixels are valid
            rasters["valid_mask_raw"] = torch.ones_like(rasters["depth_gt"]).bool()
            rasters["valid_mask_filled"] = torch.ones_like(rasters["depth_gt"]).bool()

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other
    
