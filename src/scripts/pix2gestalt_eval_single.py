import os
from tqdm import tqdm

import argparse
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

import gc
import cv2
import torch
import pix2gestalt.inference
from pix2gestalt.ldm.models.diffusion.ddim import DDIMSampler
from pix2gestalt.inference import SamPredictor, get_sam_predictor, run_inference, run_sam, load_model_from_config, run_pix2gestalt
from omegaconf import OmegaConf

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
from transformers import pipeline

def show_anns(anns):
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    if len(anns) == 0:
        return img
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img
    
def extract_mask(anns):
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    collection = []
    for ann in sorted_anns:
        m = ann['segmentation']
        collection.append(m)
    return collection


def get_sam():
    sam = sam_model_registry["vit_h"](checkpoint="./pix2gestalt/ckpt/sam_vit_h.pth")
    sam = sam.cuda()
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

def process_input(input_im):
    normalized_image = torch.from_numpy(input_im).float().permute(2, 0, 1) / 255. # [0, 255] to [0, 1]
    normalized_image = normalized_image * 2 - 1 # [0, 1] to [-1, 1]
    return normalized_image.unsqueeze(0)


def load_im(fp):
    assert os.path.exists(fp), f"File not found: {fp}"
    im = Image.open(fp).convert('RGB').resize((518, 518))
    x = np.array(im) / 255
    x = einops.rearrange(x, 'h w c -> c h w')
    x = torch.tensor(x, dtype=torch.float32)[None]
    pixel_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    pixel_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std
    return x, im


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Infer of Pix2gestalt")
    parser.add_argument("--xxx", type=str,)
    args = parser.parse_args()
    
    # Load the model
    depth_model = DepthAnythingV2(encoder='vitg', features=384, out_channels=[1536, 1536, 1536, 1536])
    depth_model.load_state_dict(torch.load('./work_dir/ckp/depth_anything_v2_vitg.pth', map_location='cpu'), strict=False)
    # depth_model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    # depth_model.load_state_dict(torch.load('./work_dir/ckp/depth_anything_v2_vitl.pth', map_location='cpu'), strict=False)
    
    depth_model.cuda().eval()

    # prepare path
    string_case = 'case3'
    amodal_path = '/ibex/ai/home/liz0l/codes/depth-fm/work_dir/teaser_save/compare/{}/pix/invisible_mask.png'.format(string_case)
    rec_path = '/ibex/ai/home/liz0l/codes/depth-fm/work_dir/teaser_save/compare/{}/pix/pred_reconstructions.jpg'.format(string_case)
    raw_depth_path = '/ibex/ai/home/liz0l/codes/depth-fm/work_dir/teaser_save/compare/{}/pix/depth_raw.png'.format(string_case)
    visible_mask_path = '/ibex/ai/home/liz0l/codes/depth-fm/work_dir/teaser_save/compare/{}/pix/visible_mask.png'.format(string_case)
    
    output_pred_path = '/ibex/ai/home/liz0l/codes/depth-fm/work_dir/teaser_save/compare/{}/pix/pred_depth.png'.format(string_case)
    output_combined_path = '/ibex/ai/home/liz0l/codes/depth-fm/work_dir/teaser_save/compare/{}/pix/combined_depth.png'.format(string_case)
    
    visible_mask = Image.open(visible_mask_path).resize((256, 256))  # [H, W, rgb]
    visible_mask = np.asarray(visible_mask) > 0
    

    # load depth gt
    gt_depth_file_path = os.path.join(raw_depth_path)
    gt_depth = Image.open(gt_depth_file_path).resize((256, 256))  # [H, W, rgb]
    gt_depth = np.asarray(gt_depth)
    gt_depth = gt_depth / 65535.0


    whole_object_rgb = Image.open(rec_path).resize((256, 256))  # [H, W, rgb]
    whole_object_rgb = np.asarray(whole_object_rgb)
    whole_object_rgb = torch.tensor(whole_object_rgb).float()
    whole_object_rgb = whole_object_rgb.unsqueeze(dim=0).permute(0, 3, 1, 2)
    whole_object_rgb = F.interpolate(whole_object_rgb, (266, 266), mode="bilinear")
    
    # preprocess rgb
    whole_object_rgb = whole_object_rgb / 255
    pixel_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    pixel_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    whole_object_rgb = (whole_object_rgb - pixel_mean) / pixel_std
    depth = depth_model(whole_object_rgb.cuda()).unsqueeze(dim=1).detach().cpu()
    
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = F.interpolate(depth, (256, 256), mode="nearest")
    depth = depth.squeeze()

    depth_save = (depth.numpy() * 65535.0).astype(np.uint16)
    Image.fromarray(depth_save).save(output_pred_path, mode="I;16")\
    
    # load invisible mask (with 512x512)
    visible_mask = visible_mask_path
    visible_mask = Image.open(visible_mask).resize((256, 256))  # [H, W, rgb]
    visible_mask = np.asarray(visible_mask) > 0
    visible_mask = visible_mask[:, :, 0]
    
    depth_align, scale, shift = align_depth_least_square(
                            gt_arr=gt_depth,
                            pred_arr=depth,
                            valid_mask_arr=torch.tensor(visible_mask),
                            return_scale_shift=True,
                            max_resolution=None)
    
    depth_align_save = (depth_align.numpy() * 65535.0).astype(np.uint16)
    Image.fromarray(depth_align_save).save(output_combined_path, mode="I;16")
    
