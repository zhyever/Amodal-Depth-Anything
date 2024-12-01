import os
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch
import matplotlib
matplotlib.use('Agg')

import torch.nn.functional as F
from src.models.amodalsynthdrive.depth_anything_v2_raw.dpt import DepthAnythingV2
from src.util.image_util import chw2hwc, colorize_depth_maps
from src.util.alignment import align_depth_least_square

from src.util import metric
from src.util.metric import MetricTracker
import einops

from src.util.logging_util import tb_logger, eval_dic_to_text
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)

from src.models import get_model
from torchvision.transforms import InterpolationMode, Resize

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Infer of Pix2gestalt")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_marigold.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--trained_checkpoint",
        type=str,
        default="./",
        help="Checkpoint path or hub name.",
    )
    args = parser.parse_args()
    
    cfg = recursive_load_config(args.config)
    model = get_model(cfg.model.name, **cfg.model.kwargs)
    model = model.from_pretrained(args.trained_checkpoint, strict=True)
    model.eval()
    model.cuda()
    
    # prepare path
    image_path = '/ibex/ai/home/liz0l/codes/Marigold/data/sam/SA-1B-Downloader/images' # raw image
    occ_image_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/occlusion'
    visible_mask_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/visible_object_mask'
    whole_mask_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/whole_mask' # gt amodal mask
    whole_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/whole'
    combined_depth_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/depth_da_update_combine' # gt depth (aligned)
    occ_depth_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/depth_da_update_occ' # depth with occ
    custom_amodal_mask_path = 'work_dir/project_folder/cvpr_base_pix2gestalt_results/amodal_mask'
    # custom_amodal_mask_path = whole_mask_path
    
    # split file
    image_filenames = os.listdir(image_path)
    with open('data_split/sam/val_80.txt', 'r') as f:
        valid_samples = f.readlines()
        valid_samples = [sample.strip() for sample in valid_samples]

    output_path = './work_dir/project_folder/cvpr_dav2/vitl_results'
    output_amodal_depth = os.path.join(output_path, 'amodal_depth')
    os.makedirs(output_amodal_depth, exist_ok=True)
    
    resize_to_hw = [518, 518]
    resize_transform = Resize(size=resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT)
    
    for image_file in tqdm(valid_samples):
        id = image_file.split("_")[1].split(".")[0]
        
        # load combined image
        image_file_path = os.path.join(occ_image_path, "{}_occlusion.png".format(id))
        image = Image.open(image_file_path) # 256
        image = np.transpose(image, (2, 0, 1)).astype(int)
        image = torch.from_numpy(np.asarray(image))
        image = image / 255
        image = resize_transform(image) # 518
        

        # load invisible mask
        visible_mask = os.path.join(visible_mask_path, "{}_visible_mask.png".format(id))
        visible_mask = Image.open(visible_mask)  #256
        visible_mask = torch.from_numpy(np.asarray(visible_mask)).unsqueeze(0)
        visible_mask = resize_transform(visible_mask) # 518

        # load depth gt
        gt_depth_file_path = os.path.join(combined_depth_path, "{}_depth.png".format(id))
        gt_depth = Image.open(gt_depth_file_path) # 512
        gt_depth = torch.from_numpy(np.asarray(gt_depth)).unsqueeze(0) 
        gt_depth = gt_depth / 65535
        gt_depth = resize_transform(gt_depth) # 518
        
        # load depth observation
        observation_depth_file_path = os.path.join(occ_depth_path, "{}_depth.png".format(id))
        observation_depth = Image.open(observation_depth_file_path) # 256
        observation_depth = torch.from_numpy(np.asarray(observation_depth)).unsqueeze(0)
        observation_depth = observation_depth / 65535
        observation_depth = resize_transform(observation_depth) # 518
        # Image.fromarray(depth_colored).save('./work_dir/debug.png')

        # use gt amodal mask to cal metric
        whole_mask = os.path.join(whole_mask_path, "{}_whole_mask.png".format(id))
        whole_mask = Image.open(whole_mask) 
        whole_mask = torch.from_numpy(np.asarray(whole_mask)).unsqueeze(0)
        whole_mask = resize_transform(whole_mask) # 518
        whole_mask = whole_mask > 0
        
        input_image = image.cuda()
        depth = model(
                    input_image.unsqueeze(dim=0), 
                    guide_rgb=None, 
                    guide_mask=(whole_mask.unsqueeze(dim=0).float().cuda() * 2) - 1,
                    observation=(observation_depth.unsqueeze(dim=0).float().cuda() * 2) - 1) # both: from -1 to 1

        depth = depth.squeeze().detach().cpu()
        depth_save = (depth.numpy() * 65535.0).astype(np.uint16)
        Image.fromarray(depth_save).save('{}/{}_depth.png'.format(output_amodal_depth, id), mode="I;16")
        
        

    