import sys
sys.path.append("pix2gestalt")
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import torch
import copy
import os
import numpy as np
import gradio as gr
import torch.nn.functional as F
from pix2gestalt.inference import load_model_from_config, run_pix2gestalt
from omegaconf import OmegaConf
from torchvision.transforms import InterpolationMode, Resize
from src.util.image_util import chw2hwc, colorize_depth_maps
from src.models.amodalsynthdrive.depth_anything_v2_raw.dpt import DepthAnythingV2
from src.models import get_model
from transformers import pipeline
from PIL import Image


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

def get_sam():
    sam = sam_model_registry["vit_h"](checkpoint="./work_dir/ckp/pix2gestalt/sam_vit_h.pth")
    sam = sam.cuda()
    mask_generator = SamAutomaticMaskGenerator(sam)
    mask_predictor = SamPredictor(sam)
    return mask_generator, mask_predictor

def load_models():
    model_raw = DepthAnythingV2(features=384, out_channels=[1536, 1536, 1536, 1536])
    # model_raw.load_state_dict(torch.load('./ckpt/depth_anything_v2_vitg.pth', map_location='cpu'), strict=False)
    model_raw.load_state_dict(torch.load('./work_dir/ckp/amodal_depth_anything_base.pth', map_location='cpu'), strict=False)
    model_raw.cuda().eval()

    depth_amodal_model = get_model('AmodalDAv2', encoder='vitl', pretrained=False)
    # depth_amodal_model = depth_amodal_model.from_pretrained('ckpt/', strict=True)
    depth_amodal_model = depth_amodal_model.from_pretrained('Zhyever/Amodal-Depth-Anything-DAV2', strict=True)
    depth_amodal_model = depth_amodal_model.cuda()
    depth_amodal_model.eval()

    ckpt = './work_dir/ckp/pix2gestalt/epoch=000005.ckpt'
    # ckpt = "./pix2gestalt/ckpt/epoch=000005.ckpt"
    config = "./pix2gestalt/configs/sd-finetune-pix2gestalt-c_concat-256.yaml"
    config = OmegaConf.load(config)
    model_gestalt = load_model_from_config(config, ckpt, f"cuda:0")
    
    pipe_matting = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    return model_raw, depth_amodal_model, model_gestalt, pipe_matting

# Base depth prediction
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

def get_points_from_components(mask, small_component_thresh=100, grid_step=10):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    selected_points = []
    
    for i in range(1, num_labels):  # Start from 1, ignoring the background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < small_component_thresh:
            # For small components, use centroid
            cX, cY = int(centroids[i][0]), int(centroids[i][1])
            selected_points.append([cX, cY])
        else:
            # For large components, select points using a grid
            component_mask = (labels == i).astype(np.uint8)
            y_coords, x_coords = np.where(component_mask)
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            for y in range(min_y, max_y, grid_step):
                for x in range(min_x, max_x, grid_step):
                    if component_mask[y, x] > 0:  # Ensure the point is inside the component
                        selected_points.append([x, y])
    
    return np.array(selected_points)

def predict_reconstructions(mask, input_image_gestalt):
    visible_mask = 255 * mask.squeeze().cpu().numpy().astype(np.uint8)
    visible_mask = cv2.resize(visible_mask, (256, 256), interpolation=cv2.INTER_AREA)
    selected_points = get_points_from_components(visible_mask)

    mask_predictor.set_image(input_image_gestalt)
    masks, _, _ = mask_predictor.predict(
    point_coords=selected_points,
    point_labels=[1] * len(selected_points),
    multimask_output=False)
    visible_mask = masks[0].astype(np.uint8) * 255

    rgb_visible_mask = np.zeros((visible_mask.shape[0], visible_mask.shape[1], 3))
    rgb_visible_mask[:, :, 0] = visible_mask
    rgb_visible_mask[:, :, 1] = visible_mask
    rgb_visible_mask[:, :, 2] = visible_mask # (256, 256, 3)
    
    cv2.imwrite('work_dir/teaser_save/sam_output.png', rgb_visible_mask)

    pred_reconstructions = run_pix2gestalt(
        model_gestalt, 'cuda', input_image_gestalt, rgb_visible_mask,
        scale = 1.5, n_samples = 1, ddim_steps = 100)

    return pred_reconstructions, visible_mask>0, selected_points

# Amodal depth prediction
def predict_dav2_amodal_depth(preprocessed_image, mask_vis, mask_type, amodal_mask_with_sam = False):
    base_depth, depth_colored_raw_hwc = predict_base_depth(preprocessed_image)
    depth_colored_raw_hwc = cv2.resize(depth_colored_raw_hwc, (preprocessed_image.shape[1], preprocessed_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask_vis is None or torch.nonzero(mask_vis).size(0) == 0:
        return depth_colored_raw_hwc, depth_colored_raw_hwc
    
    if mask_type == 'prompt_points':
        pred_reconstructions, mask, selected_points = predict_reconstructions(mask_vis, cv2.resize(preprocessed_image, (256, 256), interpolation=cv2.INTER_AREA))
    
        pred_bgr = cv2.cvtColor(pred_reconstructions[0], cv2.COLOR_RGB2BGR)
        # cv2.imwrite('work_dir/teaser_save/pred_reconstructions.jpg', pred_bgr)

        if amodal_mask_with_sam:
            # NOTE: get amodal mask with SAM
            pp_masks = mask_generator.generate(pred_reconstructions[0])
            #pp_select_mask = 1
            sorted_anns = sorted(pp_masks, key=(lambda x: x['area']), reverse=True)
            amodal_mask = None
            threshold_for_relevance = 0
            #pp_seg_masks = []
            #for ann in sorted_anns:
            #    m = ann['segmentation']
            #    pp_seg_masks.append(m)
            #amodal_mask = pp_seg_masks[pp_select_mask].astype(np.uint8)
            for ann in sorted_anns:
                mask_ = ann['segmentation'].astype(np.uint8)
                points_in_mask = 0
                for point in selected_points:
                    x, y = point
                    if mask_[y, x] == 1:
                        points_in_mask += 1
                if points_in_mask > threshold_for_relevance:
                    if amodal_mask is None:
                        amodal_mask = mask_
                    else:
                        amodal_mask = np.logical_or(amodal_mask, mask_).astype(np.uint8)
        else:
            # NOTE: get amodal mask with RMBG
            whole_object_rgb_pil = Image.fromarray(pred_reconstructions[0])
            amodal_mask = pipe_matting(whole_object_rgb_pil, return_mask=True)
            amodal_mask = np.asarray(amodal_mask)
            amodal_mask = torch.tensor(amodal_mask).float()
            amodal_mask = amodal_mask.squeeze().cpu().numpy().astype(np.uint8)
        
        guide_rgb = torch.tensor(pred_reconstructions[0]).float().unsqueeze(dim=0).permute(0, 3, 1, 2)
        guide_rgb = F.interpolate(guide_rgb, size=(518, 518), mode='bilinear')
        guide_rgb_norm = (guide_rgb / 255) * 2 - 1
    else:
        amodal_mask = mask_vis.squeeze().cpu().numpy().astype(np.uint8)
        
    amodal_mask_save = (amodal_mask * 65535.0).astype(np.uint16)
    # Image.fromarray(amodal_mask_save).save('work_dir/teaser_save/invisible_mask.png', mode="I;16")
    
    amodal_mask_ts = torch.tensor(amodal_mask)
    rgb_ts = torch.tensor(preprocessed_image).unsqueeze(dim=0).permute(0, 3, 1, 2) / 255
    resize_transform = Resize(size=(518, 518), interpolation=InterpolationMode.NEAREST)
    rgb_ts = resize_transform(rgb_ts)
    amodal_mask_ts = resize_transform(amodal_mask_ts.unsqueeze(dim=0).unsqueeze(dim=0))
    pp_amodal_mask_ts = (amodal_mask_ts > 0).float()
    # pred = depth_amodal_model(
    #             rgb_ts.float().cuda(), 
    #             guide_rgb=guide_rgb_norm.float().cuda(), 
    #             guide_mask=(pp_amodal_mask_ts.float().cuda() * 2) - 1,
    #             observation=(base_depth.unsqueeze(dim=0).unsqueeze(dim=0).cuda() * 2) - 1,
    # )
    pred = depth_amodal_model(
                rgb_ts.float().cuda(), 
                guide_rgb=None, 
                guide_mask=(pp_amodal_mask_ts.float().cuda() * 2) - 1,
                observation=(base_depth.unsqueeze(dim=0).unsqueeze(dim=0).cuda() * 2) - 1,
    )
    pred = pred.detach().cpu()

    depth_raw_np_post = F.interpolate(base_depth.squeeze().unsqueeze(dim=0).unsqueeze(dim=0), (518, 518)).squeeze()
    depth_raw_np_post_save = (depth_raw_np_post.numpy() * 65535.0).astype(np.uint16)
    # Image.fromarray(depth_raw_np_post_save).save('/ibex/ai/home/liz0l/codes/depth-fm/work_dir/teaser_save/depth_raw.png', mode="I;16")
    depth_amodal_post = F.interpolate(pred.detach().cpu().squeeze().unsqueeze(dim=0).unsqueeze(dim=0), (518, 518)).squeeze()
    depth_agg = copy.deepcopy(depth_raw_np_post)
    amodal_mask = F.interpolate(torch.tensor(amodal_mask).squeeze().unsqueeze(dim=0).unsqueeze(dim=0), (518, 518)).squeeze().cpu().numpy()
    amodal_mask = (amodal_mask > 0).astype(np.uint8) * 255
    #invisible_mask = np.logical_xor(mask, amodal_mask)
    #invisible_mask = remove_small_noise(invisible_mask)
    #kernel = np.ones((3, 3), np.uint8)
    #invisible_mask = cv2.erode(invisible_mask, kernel, iterations=1)
    #invisible_mask = cv2.morphologyEx(invisible_mask, cv2.MORPH_CLOSE, kernel)
    
    
    if mask_type == 'prompt_points':
        mask = F.interpolate(torch.tensor(mask.astype(np.uint8)).squeeze().unsqueeze(dim=0).unsqueeze(dim=0), (518, 518)).squeeze().cpu().numpy()
        depth_amodal_post = linear_regression_predict(depth_amodal_post, depth_agg, mask)

    depth_agg = median_filter_blend(depth_amodal_post, depth_agg, amodal_mask/255)

    #depth_agg[invisible_mask > 0] = depth_amodal_post[invisible_mask > 0]
    depth_agg_np = depth_agg.numpy()
    depth_agg_np_save = (depth_agg_np * 65535.0).astype(np.uint16)
    # Image.fromarray(depth_agg_np_save).save('/ibex/ai/home/liz0l/codes/depth-fm/work_dir/teaser_save/depth_agg.png', mode="I;16")
    depth_agg_colored = colorize_depth_maps(depth_agg_np, 0, 1, cmap='Spectral_r').squeeze()  # [3, H, W], value in (0, 1)
    depth_agg_colored = (depth_agg_colored * 255).astype(np.uint8)
    depth_agg_colored_hwc = chw2hwc(depth_agg_colored)
    depth_agg_colored_hwc = highlight_target(depth_agg_colored_hwc, amodal_mask)
    depth_agg_colored_hwc = cv2.resize(depth_agg_colored_hwc, (preprocessed_image.shape[1], preprocessed_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return depth_colored_raw_hwc, depth_agg_colored_hwc

def set_image(img):
    return img["background"][:,:,:3]

def process_mask(mask):
    processed_mask = torch.from_numpy(mask).float()
    return processed_mask.unsqueeze(0).unsqueeze(0)

def extract_mask(img):
    img_resized_bg = cv2.resize(img["background"], (518, 518), interpolation=cv2.INTER_AREA)
    img_resized_comp = cv2.resize(img["composite"], (518, 518), interpolation=cv2.INTER_AREA)
    img_bg_tensor = torch.tensor(img_resized_bg)
    img_comp_tesnor = torch.tensor(img_resized_comp)
    diff = torch.sum(img_bg_tensor - img_comp_tesnor, dim=-1)
    mask = diff > 0
    mask = mask.numpy()
    mask = process_mask(mask)
    return mask

def linear_regression_predict(depth_amodal_post, depth_agg, mask):
    selected_amodal = depth_amodal_post[mask > 0]
    selected_agg = depth_agg[mask > 0]   
    X_mean = selected_amodal.mean()
    y_mean = selected_agg.mean()
    numerator = torch.sum((selected_amodal - X_mean) * (selected_agg - y_mean))
    denominator = torch.sum((selected_amodal - X_mean) ** 2)

    if denominator == 0:
        raise ValueError("Denominator in slope calculation is zero.")

    slope = numerator / denominator
    intercept = y_mean - (slope * X_mean)
    depth_amodal_post_flat = depth_amodal_post.reshape(-1)
    predicted_depth = slope * depth_amodal_post_flat + intercept
    predicted_depth_map = predicted_depth.reshape(depth_amodal_post.shape)
    return predicted_depth_map

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

def remove_small_noise(binary_mask, min_area=500):
    binary_mask = (binary_mask * 255).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=4)
    cleaned_mask = np.zeros_like(binary_mask)
    
    for label in range(1, num_labels):  # Start from 1 to skip the background
        area = np.sum(labels == label)
        if area >= min_area:
            cleaned_mask[labels == label] = 255

    return cleaned_mask // 255

def setup_gradio_interface():
    title = "# Amodal Depth Anything"
    description = """Official demo for **Amodal Depth Anything: Amodal Depth Estimation in the Wild**.
    Please refer to our [paper](), [project page](), and [github]() for more details."""

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)
        original_image = gr.State(value=None)
        mask_guide = gr.State(value=None)

        with gr.Row():
            with gr.Column():
                edit_img_guide = gr.ImageEditor(
                    type="numpy",
                    layers=False,
                    brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                    eraser=False,
                )
                mask_type = gr.Dropdown(["prompt_points", "amodal_mask"], value="prompt_points", label="mask_type")
                    
            base_depth_color = gr.Image(label="Base Depth")
            agg_depth = gr.Image(label="Amodal Depth")

        amodal_depth_button = gr.Button(value='Compute Amodal Depth', interactive=True)
        
        edit_img_guide.upload(set_image, inputs=[edit_img_guide], outputs=[original_image])
        edit_img_guide.change(extract_mask, inputs=[edit_img_guide], outputs=[mask_guide])
        amodal_depth_button.click(
            predict_dav2_amodal_depth, 
            inputs=[original_image, mask_guide, mask_type], 
            outputs=[base_depth_color, agg_depth]
        )

        example_files = os.listdir('assets/app_examples')
        example_files.sort()
        example_files = [os.path.join('assets/app_examples', filename) for filename in example_files]
        gr.Examples(examples=example_files, inputs=[edit_img_guide])
        
        # demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
        demo.launch(share=True)

if __name__ == "__main__":
    model_raw, depth_amodal_model, model_gestalt, pipe_matting = load_models()
    mask_generator, mask_predictor = get_sam()
    setup_gradio_interface()