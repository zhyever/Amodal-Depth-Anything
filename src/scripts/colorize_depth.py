import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import matplotlib
import copy

def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def colorize(
    value, 
    vmin=None, 
    vmax=None, 
    cmap='turbo_r', 
    invalid_val=-99, 
    invalid_mask=None, 
    background_color=(128, 128, 128, 255), 
    gamma_corrected=False, 
    value_transform=None, 
    vminp=2, 
    vmaxp=95):
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

def highlight_target(depth_colored_hwc, mask, alpha=0.8):
    mask_3channel = cv2.merge([mask, mask, mask])
    fg_color = (200, 200, 200)  # Example gray value in BGR format
    fg_overlay = np.full_like(depth_colored_hwc, fg_color, dtype=np.uint8)
    inverse_mask = cv2.bitwise_not(mask)
    highlighted_image = np.where(mask_3channel == 0,
                                 (1 - alpha) * depth_colored_hwc + alpha * fg_overlay,
                                 depth_colored_hwc).astype(np.uint8)
    
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    highlighted_image = cv2.drawContours(highlighted_image, contours, -1, (0, 0, 0), 1)
    return highlighted_image


file_path = '/ibex/ai/home/liz0l/codes/depth-fm/work_dir/project_folder/cvpr_baseline/cvpr_base_pix2gestalt_results_update/amodal_depth/10072253_depth.png' # {}_depth.png
save_path = '/ibex/ai/home/liz0l/codes/depth-fm/work_dir/depth.png'
# stitch part1

stitch_depth = Image.open(file_path).resize((256, 256))
stitch_depth = np.asarray(stitch_depth)
stitch_depth_colored = colorize(stitch_depth, cmap='Spectral_r')[:, :, :3]
cv2.imwrite(save_path, stitch_depth_colored[:, :, [2, 1, 0]])
