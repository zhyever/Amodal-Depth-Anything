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
from torchvision.transforms import InterpolationMode, Resize

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


def colorize(value, vmin=None, vmax=None, cmap='turbo_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None, vminp=2, vmaxp=95):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
        
    mask = np.logical_not(invalid_mask)

    # normalize
    # vmin = np.percentile(value[mask],2) if vmin is None else vmin
    # vmax = np.percentile(value[mask],85) if vmax is None else vmax
    
    # if vminp is None:
    #     vmin = value.min()
    # else:
    vmin = np.percentile(value[mask],vminp) if vmin is None else vmin
    vmax = np.percentile(value[mask],vmaxp) if vmax is None else vmax
    
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Infer of Pix2gestalt")
    parser.add_argument("--xxx", type=str,)
    args = parser.parse_args()
    
    # load gestalt model
    # ckpt="./pix2gestalt/ckpt/epoch=000005.ckpt"
    # config="./pix2gestalt/configs/sd-finetune-pix2gestalt-c_concat-256.yaml"
    # device_idx = '0'
    # device = f"cuda:{device_idx}"
    # config = OmegaConf.load(config)
    # model = load_model_from_config(config, ckpt, device)
    
    # Load the model
    # depth_model = DepthAnythingV2(encoder='vitg', features=384, out_channels=[1536, 1536, 1536, 1536])
    # depth_model = DepthAnythingV2(encoder='vitl', features=384, out_channels=[1536, 1536, 1536, 1536])
    
    # depth_model.load_state_dict(torch.load('./work_dir/ckp/depth_anything_v2_vitg.pth', map_location='cpu'), strict=False)
    # depth_model.cuda().eval()

    # prepare path
    image_path = '/ibex/ai/home/liz0l/codes/Marigold/data/sam/SA-1B-Downloader/images' # raw image
    occ_image_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/occlusion'
    visible_mask_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/visible_object_mask'
    whole_mask_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/whole_mask'
    whole_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/whole'
    combined_depth_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/depth_da_update_combine' # gt
    occ_depth_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/depth_da_update_occ'
    
    # split file
    image_filenames = os.listdir(image_path)
    with open('data_split/sam/val_80.txt', 'r') as f:
        valid_samples = f.readlines()
        valid_samples = [sample.strip() for sample in valid_samples]

    # eval
    eval_metrics = ['rmse_linear', 'log10', 'delta1_acc']
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]
    val_align_easy_metrics = MetricTracker(*[m.__name__ for m in metric_funcs])
    val_align_mid_metrics = MetricTracker(*[m.__name__ for m in metric_funcs])
    val_align_diff_metrics = MetricTracker(*[m.__name__ for m in metric_funcs])
    val_align_metrics = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_dict = {
        "align_easy": val_align_easy_metrics,
        "align_mid": val_align_mid_metrics,
        "align_diff": val_align_diff_metrics,
        "align_overall": val_align_metrics,}
    
    output_path = './work_dir/project_folder/cvpr_baseline/cvpr_base_pix2gestalt_results'
    predicted_depth_path = os.path.join(output_path, 'amodal_depth')
    predicted_amodal_path = os.path.join(output_path, 'amodal_mask')
    
    
    resize_to_hw = [518, 518]
    resize_transform = Resize(size=resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT)
    
    for image_file in tqdm(valid_samples):
        id = image_file.split("_")[1].split(".")[0]

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
        observation_file_path = os.path.join(occ_depth_path, "{}_depth.png".format(id))
        observation = Image.open(observation_file_path) # 512
        observation = torch.from_numpy(np.asarray(observation)).unsqueeze(0) 
        observation = observation / 65535
        observation = resize_transform(observation) # 518
        
        # load depth pred
        pred_depth_file_path = os.path.join(predicted_depth_path, "{}_depth.png".format(id))
        pred_depth = Image.open(pred_depth_file_path) # 512
        pred_depth = torch.from_numpy(np.asarray(pred_depth)).unsqueeze(0) 
        pred_depth = pred_depth / 65535
        pred_depth = resize_transform(pred_depth) # 518

        
        depth_align, scale, shift = align_depth_least_square(
                                gt_arr=observation,
                                pred_arr=pred_depth,
                                valid_mask_arr=torch.tensor(visible_mask),
                                return_scale_shift=True,
                                max_resolution=None)
        
        # metric
        # check difficulty
        # use gt amodal mask to cal metric
        whole_mask = os.path.join(whole_mask_path, "{}_whole_mask.png".format(id))
        whole_mask = Image.open(whole_mask) 
        whole_mask = torch.from_numpy(np.asarray(whole_mask)).unsqueeze(0)
        whole_mask = resize_transform(whole_mask) # 518
        whole_mask = whole_mask > 0
        
        object_mask = whole_mask
        visibility_mask = torch.tensor(visible_mask)
        image_size_h, image_size_w = object_mask.shape[-2], object_mask.shape[-1]
        object_mask_pixel_num = torch.sum(object_mask > 0)
        visibility_mask_pixel_num = torch.sum(visibility_mask > 0)
        object_ratio = object_mask_pixel_num / (image_size_h * image_size_w)
        visibility_ratio = visibility_mask_pixel_num / object_mask_pixel_num
        
        if visibility_ratio > 0.75:
            visibility_size = 'large'
        elif visibility_ratio > 0.5:
            visibility_size = 'medium'
        else:
            visibility_size = 'small'

        if visibility_size == 'small':
            select_tracker_align = val_align_diff_metrics
        elif visibility_size == 'medium':
            select_tracker_align = val_align_mid_metrics
        elif visibility_size == 'large':
            select_tracker_align = val_align_easy_metrics
        else:
            raise NotImplementedError
        
        sample_metric = []

        # load invisible mask
        # predicted_amodal_mask = os.path.join(predicted_amodal_path, "{}_amodal_mask.png".format(id))
        # predicted_amodal_mask = Image.open(predicted_amodal_mask)  #256
        # predicted_amodal_mask = torch.from_numpy(np.asarray(predicted_amodal_mask)).unsqueeze(0)
        # predicted_amodal_mask = resize_transform(predicted_amodal_mask) # 518
        # predicted_amodal_mask = predicted_amodal_mask
        # whole_mask_insec = torch.logical_and(whole_mask, predicted_amodal_mask).squeeze().cuda()
        
        invisible_mask = torch.logical_and(torch.logical_not(visibility_mask.squeeze().cuda()), whole_mask.squeeze().cuda())
        depth_align = torch.tensor(depth_align).squeeze().cuda()
        gt_depth = torch.tensor(gt_depth).squeeze().cuda()
        
        # plt.imshow(invisible_mask)
        # plt.savefig("work_dir/debug/pix2gestalt_output_invisible.png")
        
        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_align + 1e-5, gt_depth + 1e-5, invisible_mask)
            gather_metric = [_metric.cuda()]
            sample_metric.append(_metric.__str__())
            for m in gather_metric:
                if torch.isnan(m).any():
                    continue # skip nan case
                val_align_metrics.update(_metric_name, m.item())
                select_tracker_align.update(_metric_name, m.item())
           
    return_dict = {
        "align_easy": val_align_easy_metrics.result(),
        "align_mid": val_align_mid_metrics.result(),
        "align_diff": val_align_diff_metrics.result(),
        "align_overall": val_align_metrics.result(),}
    

    for metric_k, metric_v in return_dict.items():
        text = eval_dic_to_text(
                val_metrics=metric_v,
                dataset_name='sam-pix2genstalt',
                sample_list_path='',
                diff=metric_k,)
        print(text)
        with open(os.path.join('./work_dir/project_folder/cvpr_baseline/cvpr_base_pix2gestalt_results_update', "eval_result_gtamodal.txt"), "a") as f:
            f.write(text)
            f.write("\n")