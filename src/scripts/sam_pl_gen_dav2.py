
import os
from PIL import Image
# from marigold import MarigoldPipeline
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

import os
import torch
import einops
import argparse
import numpy as np
from PIL import Image
from PIL.Image import Resampling
import matplotlib.pyplot as plt

from src.models.amodalsynthdrive.depth_anything_v2_raw.dpt import DepthAnythingV2
import torch.nn.functional as F
from src.util.image_util import chw2hwc, colorize_depth_maps

from src.util.alignment import align_depth_least_square
import copy

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

parser = argparse.ArgumentParser()
parser.add_argument("--data_index", type=int,)
args = parser.parse_args()

image_path = '/ibex/ai/home/liz0l/codes/Marigold/data/sam/SA-1B-Downloader/images' # raw image
occ_image_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/occlusion'
visible_mask_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/visible_object_mask'
whole_mask_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/whole_mask'
whole_path = '/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/whole'



image_filenames = os.listdir(image_path)
with open('/ibex/ai/home/liz0l/codes/Marigold/data/sam/valid.txt', 'r') as f:
    valid_samples = f.readlines()
    valid_samples = [sample.strip() for sample in valid_samples]
chunk_size = 40000
num_chunks = np.ceil(len(valid_samples) / chunk_size).astype(int) # 12 index: 0-11
chunks = np.array_split(valid_samples, num_chunks)
valid_samples = chunks[args.data_index]


# Load the model
model = DepthAnythingV2(encoder='vitg', features=384, out_channels=[1536, 1536, 1536, 1536])
model.load_state_dict(torch.load('./work_dir/ckp/depth_anything_v2_vitg.pth', map_location='cpu'), strict=False)
model.cuda().eval()
        
for image_file in tqdm(valid_samples):
    image_file_path = os.path.join(image_path, "sa_{}.jpg".format(image_file))

    # Load an image
    im, im_show = load_im(image_file_path)
    im = im.cuda()

    # Generate depth
    depth = model(im).unsqueeze(dim=1).detach().cpu()
    depth = F.interpolate(depth, (518, 518), mode="nearest")
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = depth.squeeze()
    depth_save = depth.numpy().astype(np.uint16) 
    
    
    occ_file_path = os.path.join(occ_image_path, "{}_occlusion.png".format(image_file))
    
    # Load an image
    im, im_show = load_im(occ_file_path)
    im = im.cuda()

    # Generate depth
    occ_depth = model(im).unsqueeze(dim=1).detach().cpu()
    occ_depth = F.interpolate(occ_depth, (518, 518), mode="nearest")
    occ_depth = (occ_depth - occ_depth.min()) / (occ_depth.max() - occ_depth.min())
    occ_depth = occ_depth.squeeze()
    occ_depth_save = (occ_depth.numpy() * 65535.0).astype(np.uint16)
    
    visible_mask = os.path.join(visible_mask_path, "{}_visible_mask.png".format(image_file))
    visible_mask = Image.open(visible_mask).resize((518, 518))  # [H, W, rgb]
    visible_mask = np.asarray(visible_mask) > 0
    
    whole_mask = os.path.join(whole_mask_path, "{}_whole_mask.png".format(image_file))
    whole_mask = Image.open(whole_mask).resize((518, 518))  # [H, W, rgb]
    whole_mask = np.asarray(whole_mask) > 0
    
    
    depth_align, scale, shift = align_depth_least_square(
                                gt_arr=occ_depth,
                                pred_arr=depth,
                                valid_mask_arr=torch.tensor(visible_mask),
                                return_scale_shift=True,
                                max_resolution=None)
    
    # depth_align, scale, shift = align_depth_least_square(
    #                             gt_arr=occ_depth,
    #                             pred_arr=depth,
    #                             valid_mask_arr=torch.logical_not(torch.tensor(whole_mask)),
    #                             return_scale_shift=True,
    #                             max_resolution=None)
    
    combine_depth = torch.clone(occ_depth)
    combine_depth[torch.tensor(whole_mask)] = (depth * scale + shift)[torch.tensor(whole_mask)]
    combine_depth_save = (combine_depth.numpy() * 65535.0).astype(np.uint16)
    
    
    # Image.fromarray(occ_depth_save).save('/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/depth_da_update_occ/{}_depth.png'.format(image_file), mode="I;16")
    Image.fromarray(combine_depth_save).resize((512, 512)).save('/ibex/ai/home/liz0l/codes/depth-fm/data/sam/pix2gestalt_occlusions_release/depth_da_update_combine/{}_depth.png'.format(image_file), mode="I;16")
    
    # plt.subplot(2, 4, 1)
    # plt.imshow(im_show)
    # plt.subplot(2, 4, 2)
    # depth_colored = colorize_depth_maps(depth.squeeze().detach().cpu().numpy(), 0, 1, cmap='Spectral').squeeze()  # [3, H, W], value in (0, 1)
    # depth_colored = (depth_colored * 255).astype(np.uint8)
    # depth_colored_hwc = chw2hwc(depth_colored)
    # plt.imshow(depth_colored_hwc)
    # plt.subplot(2, 4, 3)
    # depth_colored = colorize_depth_maps(occ_depth.squeeze().detach().cpu().numpy(), 0, 1, cmap='Spectral').squeeze()  # [3, H, W], value in (0, 1)
    # depth_colored = (depth_colored * 255).astype(np.uint8)
    # depth_colored_hwc = chw2hwc(depth_colored)
    # plt.imshow(depth_colored_hwc)
    # plt.subplot(2, 4, 4)
    # plt.imshow(visible_mask)
    # plt.subplot(2, 4, 5)
    # plt.imshow(whole_mask)
    # plt.subplot(2, 4, 6)
    # depth_colored = colorize_depth_maps(combine_depth.squeeze().detach().cpu().numpy(), 0, 1, cmap='Spectral').squeeze()  # [3, H, W], value in (0, 1)
    # depth_colored = (depth_colored * 255).astype(np.uint8)
    # depth_colored_hwc = chw2hwc(depth_colored)
    # plt.imshow(depth_colored_hwc)
    # whole_file_path = os.path.join(whole_path, "{}_whole.png".format(image_file))
    # Load an image
    # im, im_show = load_im(whole_file_path)
    # plt.subplot(2, 4, 7)
    # plt.imshow(im_show)    
    # plt.savefig('work_dir/gen_{}_depth_part.png'.format(image_file))
    # exit(100)
    
    
        