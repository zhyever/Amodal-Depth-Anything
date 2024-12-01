
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
from depthfm import DepthFM
import matplotlib.pyplot as plt

def load_im(fp):
    assert os.path.exists(fp), f"File not found: {fp}"
    im = Image.open(fp).convert('RGB').resize((512, 512))
    x = np.array(im)
    x = einops.rearrange(x, 'h w c -> c h w')
    x = x / 127.5 - 1
    x = torch.tensor(x, dtype=torch.float32)[None]
    return x

parser = argparse.ArgumentParser()
parser.add_argument("--data_index", type=int,)
args = parser.parse_args()

image_path = '/ibex/ai/home/liz0l/codes/Marigold/data/sam/SA-1B-Downloader/images'
image_filenames = os.listdir(image_path)
with open('/ibex/ai/home/liz0l/codes/Marigold/data/sam/valid.txt', 'r') as f:
    valid_samples = f.readlines()
    valid_samples = [sample.strip() for sample in valid_samples]
chunk_size = 40000
num_chunks = np.ceil(len(valid_samples) / chunk_size).astype(int) # 12 index: 0-11
chunks = np.array_split(valid_samples, num_chunks)
valid_samples = chunks[args.data_index]


# Load the model
model = DepthFM('checkpoints/depthfm-v1.ckpt')
model.cuda().eval()
        
for image_file in tqdm(valid_samples):
    image_file_path = os.path.join(image_path, "sa_{}.jpg".format(image_file))
    # Predict depth

    # Load an image
    im = load_im(image_file_path)
    im = im.cuda()

    # Generate depth
    dtype = torch.float16
    model.model.dtype = dtype
    with torch.autocast(device_type="cuda", dtype=dtype):
        depth = model.predict_depth(im, num_steps=2, ensemble_size=10)
    depth = depth.squeeze(0).squeeze(0).cpu().numpy()

    depth_to_save = (depth * 65535.0).astype(np.uint16)
    Image.fromarray(depth_to_save).save('/ibex/ai/home/liz0l/codes/Marigold/data/sam/pix2gestalt_occlusions_release/depth/{}_depth.png'.format(image_file), mode="I;16")
    
    # plt.subplot(1, 2, 1)
    # plt.imshow(input_image)
    # plt.subplot(1, 2, 2)
    # plt.imshow(pipe_out.depth_np)
    # plt.savefig('/ibex/ai/home/liz0l/codes/Marigold/work_dir/debug/{}_depth.png'.format(image_file))
    # exit(100)
        