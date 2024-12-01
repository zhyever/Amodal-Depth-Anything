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

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

from transformers import BlipProcessor, BlipForConditionalGeneration
import src.scripts.blip_transform as T
# import GroundingDINO.groundingdino.datasets.transforms as T

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

def generate_caption(raw_image, device):
    # unconditional image captioning
    if device == "cuda":
        inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
    else:
        inputs = processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def load_image_blip(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


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
    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
        
        
    # Load the model
    depth_model = DepthAnythingV2(encoder='vitg', features=384, out_channels=[1536, 1536, 1536, 1536])
    depth_model.load_state_dict(torch.load('./work_dir/ckp/depth_anything_v2_vitg.pth', map_location='cpu'), strict=False)
    depth_model.cuda().eval()

    # prepare path
    image_path = '/ibex/ai/home/liz0l/codes/Marigold/data/sam/SA-1B-Downloader/images' # raw image
    occ_image_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/occlusion'
    visible_mask_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/visible_object_mask'
    whole_mask_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/whole_mask'
    whole_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/whole'
    combined_depth = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/depth_da_update_combine'
    object_depth = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/depth_da_update_occ' # gt
    
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
    
    output_path = './work_dir/project_folder/cvpr_base_pix2gestalt_results'
    output_amodal_mask = os.path.join(output_path, 'amodal_mask')
    output_amodal_inpainting = os.path.join(output_path, 'amodal_inpainting')
    output_amodal_depth = os.path.join(output_path, 'amodal_depth')
    output_amodal_aligned_depth = os.path.join(output_path, 'amodal_aligned_depth')
    os.makedirs(output_amodal_mask, exist_ok=True)
    os.makedirs(output_amodal_inpainting, exist_ok=True)
    os.makedirs(output_amodal_depth, exist_ok=True)
    os.makedirs(output_amodal_aligned_depth, exist_ok=True)
    
    
    for image_file in tqdm(valid_samples):
        id = image_file.split("_")[1].split(".")[0]
        
        # load combined image
        image_file_path = os.path.join(occ_image_path, "{}_occlusion.png".format(id))
        input_image = cv2.imread(image_file_path) 
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) 
        input_image_gestalt = cv2.resize(input_image, (256, 256), interpolation=cv2.INTER_AREA)

        # load invisible mask
        visible_mask_path = os.path.join(visible_mask_path, "{}_visible_mask.png".format(id))
        visible_mask = Image.open(visible_mask_path).resize((256, 256))  # [H, W, rgb]
        visible_mask = np.asarray(visible_mask) > 0
        
        # load whole mask
        whole_mask_path = os.path.join(whole_mask_path, "{}_whole_mask.png".format(id))
        whole_mask = Image.open(whole_mask_path).resize((256, 256))  # [H, W, rgb]
        # whole_mask = Image.open(whole_mask_path).resize((1024, 1024))  # [H, W, rgb]
        
        whole_mask = np.asarray(whole_mask) > 0
        whole_mask = torch.tensor(whole_mask).unsqueeze(dim=-1).repeat(1, 1, 3)
        whole_mask = whole_mask.numpy()
        
        # target area
        # occ_mask = np.logical_and(np.logical_not(visible_mask), whole_mask)
        occ_mask = visible_mask
        occ_mask = torch.tensor(occ_mask).unsqueeze(dim=-1).repeat(1, 1, 3)
        occ_mask = occ_mask.numpy()

        # load depth gt
        gt_depth_file_path = os.path.join(combined_depth, "{}_depth.png".format(id))
        gt_depth = Image.open(gt_depth_file_path).resize((256, 256))  # [H, W, rgb]
        gt_depth = np.asarray(gt_depth)
        gt_depth = gt_depth / 65535.0
        
        image_pil, image = load_image_blip(image_file_path)
        image_pil_np = np.array(image_pil)
        image_pil_np = image_pil_np * occ_mask
        occ_mask = occ_mask[:, :, 0]
        image_pil = Image.fromarray(image_pil_np)
        image_pil.save('./work_dir/sdxl_input.png')
        caption = generate_caption(image_pil, "cuda")
        
        image = load_image(image_file_path).resize((1024, 1024))
        # mask_image = load_image(visible_mask_path).resize((1024, 1024))
        # mask_image_np = np.array(mask_image)
        # mask_image_np_reverse = 255 - mask_image_np
        mask_image = Image.fromarray(occ_mask)
        prompt = caption
        generator = torch.Generator(device="cuda").manual_seed(0)

        image = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            guidance_scale=8.0,
            num_inference_steps=20,  # steps between 15 and 30 work well for us
            strength=0.99,  # make sure to use `strength` below 1.0
            generator=generator,
            ).images[0]

        image.save('./work_dir/sdxl_inference.png')
        image_np = np.array(image)
        image_ts = torch.tensor(image_np)
        print(image_np.shape)
        exit(100)

        # run pix2gestalt
        mask = visible_mask
        visible_mask = 255 * np.squeeze(mask).astype(np.uint8)
        
        rgb_visible_mask = np.zeros((visible_mask.shape[0], visible_mask.shape[1], 3))
        rgb_visible_mask[:, :, 0] = visible_mask
        rgb_visible_mask[:, :, 1] = visible_mask
        rgb_visible_mask[:, :, 2] = visible_mask # (256, 256, 3)

        pred_reconstructions = run_pix2gestalt(
            model, 'cuda', input_image_gestalt, rgb_visible_mask,
            scale = 1, n_samples = 1, ddim_steps = 200)
        
        # plt.imshow(pred_reconstructions[0])
        # plt.savefig("work_dir/debug/pix2gestalt_whole_object_rgb.png")
        Image.fromarray(pred_reconstructions[0]).save('{}/{}_amodal_inpainting.png'.format(output_amodal_inpainting, id))
        
        # extract amodal mask
        whole_object_rgb = pred_reconstructions[0]
        whole_object_rgb = torch.tensor(whole_object_rgb).float()
        whole_object_rgb_sum = whole_object_rgb[:, :, 0] + whole_object_rgb[:, :, 1] + whole_object_rgb[:, :, 2]
        amodal_mask = whole_object_rgb_sum < 250 * 3
        
        amodal_mask_da_input = F.interpolate(amodal_mask.unsqueeze(dim=0).unsqueeze(dim=0).float(), (518, 518), mode="nearest").squeeze() > 0
        amodal_mask_da_input_save = (amodal_mask_da_input.numpy() * 65535.0).astype(np.uint16)
        Image.fromarray(amodal_mask_da_input_save).save('{}/{}_amodal_mask.png'.format(output_amodal_mask, id), mode="I;16")
        
        amodal_mask = F.interpolate(amodal_mask.unsqueeze(dim=0).unsqueeze(dim=0).float(), (256, 256), mode="nearest").squeeze() > 0
        # plt.imshow(amodal_mask)
        # plt.savefig("work_dir/debug/pix2gestalt_output_mask.png")
        
        whole_object_rgb = whole_object_rgb.unsqueeze(dim=0).permute(0, 3, 1, 2)
        whole_object_rgb = F.interpolate(whole_object_rgb, (266, 266), mode="bilinear")
        
        # preprocess rgb
        whole_object_rgb = whole_object_rgb / 255
        pixel_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        pixel_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        whole_object_rgb = (whole_object_rgb - pixel_mean) / pixel_std

        # im, im_show = load_im(image_file_path)
        # amodal_mask_da_input = amodal_mask_da_input.unsqueeze(dim=0).unsqueeze(dim=0).repeat(1, 3, 1, 1).bool()
        # im[amodal_mask_da_input] = whole_object_rgb
        # im = im * ~amodal_mask_da_input + whole_object_rgb * amodal_mask_da_input
        
        depth = depth_model(whole_object_rgb.cuda()).unsqueeze(dim=1).detach().cpu()
        # depth = depth_model(im.cuda()).unsqueeze(dim=0).detach().cpu()
        
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = F.interpolate(depth, (256, 256), mode="nearest")
        depth = depth.squeeze()
        
        depth_save = (depth.numpy() * 65535.0).astype(np.uint16)
        Image.fromarray(depth_save).save('{}/{}_depth.png'.format(output_amodal_depth, id), mode="I;16")
        depth_colored = colorize(depth, vmin=0, vmax=1, cmap='Spectral', invalid_mask=np.logical_not(amodal_mask.numpy()))
        Image.fromarray(depth_colored).save('{}/{}_depth_colored.png'.format(output_amodal_depth, id))
        
        depth_colored = colorize_depth_maps(depth.squeeze().detach().cpu().numpy(), 0, 1, cmap='Spectral').squeeze()  # [3, H, W], value in (0, 1)
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        # plt.imshow(depth_colored_hwc)
        # plt.savefig("work_dir/debug/pix2gestalt_output_depth.png")
        
        # load invisible mask (with 512x512)
        visible_mask = os.path.join(visible_mask_path, "{}_visible_mask.png".format(id))
        visible_mask = Image.open(visible_mask).resize((256, 256))  # [H, W, rgb]
        visible_mask = np.asarray(visible_mask) > 0
        depth_align, scale, shift = align_depth_least_square(
                                gt_arr=gt_depth,
                                pred_arr=depth,
                                valid_mask_arr=torch.tensor(visible_mask),
                                return_scale_shift=True,
                                max_resolution=None)
        
        depth_align_save = (depth_align.numpy() * 65535.0).astype(np.uint16)
        Image.fromarray(depth_align_save).save('{}/{}_depth.png'.format(output_amodal_aligned_depth, id), mode="I;16")
        
        # depth_align[np.logical_not(amodal_mask)] = 0
        # depth_colored = colorize_depth_maps(depth_align.squeeze().detach().cpu().numpy(), 0, 1, cmap='Spectral').squeeze()  # [3, H, W], value in (0, 1)
        # depth_colored = (depth_colored * 255).astype(np.uint8)
        # depth_colored_hwc = chw2hwc(depth_colored)
        # plt.imshow(depth_colored_hwc)
        # plt.savefig("work_dir/debug/pix2gestalt_output_aligned_depth.png")
        
        # metric
        # check difficulty
        object_mask = amodal_mask
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
        

        whole_mask = os.path.join(whole_mask_path, "{}_whole_mask.png".format(id))
        whole_mask = Image.open(whole_mask).resize((256, 256))  # [H, W, rgb]
        whole_mask = np.asarray(whole_mask) > 0
        whole_mask = torch.tensor(whole_mask)
    
        invisible_mask = torch.logical_and(torch.logical_not(visibility_mask), whole_mask)
        depth_align = torch.tensor(depth_align)
        gt_depth = torch.tensor(gt_depth)
        
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
        with open(os.path.join('work_dir/project_folder/cvpr_base_pix2gestalt_results', "eval.txt"), "a") as f:
            f.write(text)
            f.write("\n")