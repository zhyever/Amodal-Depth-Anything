import os
import cv2
import copy
import torch
import argparse

import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode, Resize

from src.models import get_model
from src.models.amodalsynthdrive.depth_anything_v2_raw.dpt import DepthAnythingV2
from src.util.image_util import chw2hwc, colorize_depth_maps

def predict_base_depth(input_image_raw):
    input_image_raw = cv2.resize(input_image_raw, (518, 518))
    input_image_raw_ts = torch.tensor(input_image_raw).permute(2, 0, 1).unsqueeze(dim=0) / 255
    pp_input_image_raw_ts = (input_image_raw_ts - torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    depth_raw = model_raw(pp_input_image_raw_ts.cuda()).unsqueeze(dim=1).detach().cpu()
    depth_raw = F.interpolate(depth_raw, (518, 518), mode="nearest")
    depth_raw = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min())
    depth_raw = depth_raw.squeeze()
    depth_raw_np = depth_raw.numpy()
    depth_colored_raw = colorize_depth_maps(depth_raw_np, 0, 1, cmap='Spectral_r').squeeze()  # [3, H, W], value in (0, 1)
    depth_colored_raw = (depth_colored_raw * 255).astype(np.uint8)
    depth_colored_raw_hwc = chw2hwc(depth_colored_raw)
    return depth_raw, depth_colored_raw_hwc

def median_filter_blend(depth_amodal_post, depth_agg, mask, filter_width=3):
    mask = torch.tensor(mask, device=depth_agg.device)
    blended_depth = depth_agg.clone()
    blended_depth[mask > 0] = depth_amodal_post[mask > 0]

    kernel = torch.ones((1, 1, filter_width, filter_width), device=mask.device)
    dilated_mask = F.conv2d(mask.float().unsqueeze(0).unsqueeze(0), kernel, padding=filter_width // 2)
    border_mask = (dilated_mask > 0) & (dilated_mask < filter_width ** 2)
    border_mask = border_mask.squeeze()

    blended_depth_np = blended_depth.detach().cpu().numpy()
    median_filtered = cv2.blur(blended_depth_np, (filter_width, filter_width))

    blended_depth_np[border_mask.cpu().numpy()] = median_filtered[border_mask.cpu().numpy()]
    return torch.tensor(blended_depth_np, device=depth_agg.device)

def highlight_target(depth_colored_hwc, mask, alpha=0.0):
    mask_3channel = cv2.merge([mask, mask, mask])
    fg_color = (200, 200, 200)  # Example gray value in BGR format
    fg_overlay = np.full_like(depth_colored_hwc, fg_color, dtype=np.uint8)
    inverse_mask = cv2.bitwise_not(mask)
    highlighted_image = np.where(mask_3channel == 0,
                                 (1 - alpha) * depth_colored_hwc + alpha * fg_overlay,
                                 depth_colored_hwc).astype(np.uint8)
    
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    highlighted_image = cv2.drawContours(highlighted_image, contours, -1, (0, 0, 0), 2)
    return highlighted_image

def load_models():
    model_raw = DepthAnythingV2(features=384, out_channels=[1536, 1536, 1536, 1536])
    model_raw.load_state_dict(torch.load('work_dir/ckp/amodal_depth_anything_base.pth', map_location='cpu'), strict=False) # hg repo
    model_raw.cuda().eval()

    depth_amodal_model = get_model('AmodalDAv2', encoder='vitl', pretrained=False)
    # depth_amodal_model = depth_amodal_model.from_pretrained('work_dir/project_folder/cvpr_dav2/variants/dav2_vitl/20241023_145739/checkpoint/iter_060000', strict=True)
    depth_amodal_model = depth_amodal_model.from_pretrained('Zhyever/Amodal-Depth-Anything-DAV2', strict=True) # hg repo
    depth_amodal_model.cuda().eval()
    
    return model_raw, depth_amodal_model

def infer_single_image(input_image_path, input_mask_path, output_path, model_raw, depth_amodal_model):
    file_name = os.path.basename(input_image_path).split('.')[0]
    os.makedirs(output_path, exist_ok=True)
    
    preprocessed_image = cv2.imread(input_image_path)
    base_depth, depth_colored_raw_hwc = predict_base_depth(preprocessed_image)
    depth_colored_raw_hwc = cv2.resize(depth_colored_raw_hwc, (preprocessed_image.shape[1], preprocessed_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    amodal_mask = Image.open(input_mask_path) 
    amodal_mask = np.asarray(amodal_mask) > 0
    amodal_mask_ts = torch.tensor(amodal_mask).float()
    
    rgb_ts = torch.tensor(preprocessed_image).unsqueeze(dim=0).permute(0, 3, 1, 2) / 255
    resize_transform = Resize(size=(518, 518), interpolation=InterpolationMode.NEAREST)
    rgb_ts = resize_transform(rgb_ts)
    amodal_mask_ts = resize_transform(amodal_mask_ts.unsqueeze(dim=0).unsqueeze(dim=0))
    pp_amodal_mask_ts = (amodal_mask_ts > 0).float()
    pred = depth_amodal_model(
                rgb_ts.float().cuda(), 
                guide_rgb=None, 
                guide_mask=(pp_amodal_mask_ts.float().cuda() * 2) - 1,
                observation=(base_depth.unsqueeze(dim=0).unsqueeze(dim=0).cuda() * 2) - 1,
    )
    pred = pred.detach().cpu()
    
            
    depth_raw_np_post = F.interpolate(base_depth.squeeze().unsqueeze(dim=0).unsqueeze(dim=0), (518, 518)).squeeze()
    depth_amodal_post = F.interpolate(pred.detach().cpu().squeeze().unsqueeze(dim=0).unsqueeze(dim=0), (518, 518)).squeeze()
    depth_agg = copy.deepcopy(depth_raw_np_post)
    amodal_mask = F.interpolate(torch.tensor(amodal_mask).float().squeeze().unsqueeze(dim=0).unsqueeze(dim=0), (518, 518)).squeeze().cpu().numpy()
    amodal_mask = (amodal_mask > 0).astype(np.uint8) * 255
        
    depth_agg = median_filter_blend(depth_amodal_post, depth_agg, amodal_mask/255)

    #depth_agg[invisible_mask > 0] = depth_amodal_post[invisible_mask > 0]
    depth_agg_np = depth_agg.numpy()
    depth_agg_np_save = (depth_agg_np * 65535.0).astype(np.uint16)
    # Image.fromarray(depth_agg_np_save).save(os.path.join(output_path, '{}_xx.png'.format(file_name)), mode="I;16") # Save as 16-bit PNG
    depth_agg_colored = colorize_depth_maps(depth_agg_np, 0, 1, cmap='Spectral_r').squeeze()
    depth_agg_colored = (depth_agg_colored * 255).astype(np.uint8)
    depth_agg_colored_hwc = chw2hwc(depth_agg_colored)
    depth_agg_colored_hwc = highlight_target(depth_agg_colored_hwc, amodal_mask)
    depth_agg_colored_hwc = cv2.resize(depth_agg_colored_hwc, (preprocessed_image.shape[1], preprocessed_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    depth_colored_raw_hwc = depth_colored_raw_hwc[:, :, [2, 1, 0]]
    depth_agg_colored_hwc = depth_agg_colored_hwc[:, :, [2, 1, 0]]
    
    cv2.imwrite(os.path.join(output_path, '{}_raw_depth_rendered.png'.format(file_name)), depth_colored_raw_hwc)
    cv2.imwrite(os.path.join(output_path, '{}_amodal_depth_rendered.png'.format(file_name)), depth_agg_colored_hwc)
    
    return depth_colored_raw_hwc, depth_agg_colored_hwc
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--input_image_path",
        type=str,
        help="Path to the input image.")
    parser.add_argument(
        "--input_mask_path",
        type=str,
        help="Path to the amodal mask image.")
    parser.add_argument(
        "--output_folder", 
        type=str,
        help="Output folder.")
    args = parser.parse_args()
    
    model_raw, depth_amodal_model = load_models()
    infer_single_image(args.input_image_path, args.input_mask_path, args.output_folder, model_raw, depth_amodal_model)