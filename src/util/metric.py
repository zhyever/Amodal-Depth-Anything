# This file is modified based on the original file from the marigold repository
# Author: Zhenyu Li


import pandas as pd
import torch
from scipy import ndimage
import copy
import numpy as np
from skimage.feature import canny

# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None):
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
        )
    else:
        diff = torch.abs(torch.log10(output) - torch.log10(target))
    return diff.mean()


# adapt from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None):
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss


def eps(x):
    """Return the `eps` value for the given `input` dtype. (default=float32 ~= 1.19e-7)"""
    dtype = torch.float32 if x is None else x.dtype
    return torch.finfo(dtype).eps

def to_log(depth):
    """Convert linear depth into log depth."""
    depth = torch.tensor(depth)
    depth = (depth > 0) * depth.clamp(min=eps(depth)).log()
    return depth

def to_inv(depth):
    """Convert linear depth into disparity."""
    depth = torch.tensor(depth)
    disp = (depth > 0) / depth.clamp(min=eps(depth))
    return disp

def extract_edges(depth,
                  preprocess=None,
                  sigma=1,
                  mask=None,
                  use_canny=True):
    """Detect edges in a dense LiDAR depth map.

    :param depth: (ndarray) (h, w, 1) Dense depth map to extract edges.
    :param preprocess: (str) Additional depth map post-processing. (log, inv, none)
    :param sigma: (int) Gaussian blurring sigma.
    :param mask: (Optional[ndarray]) Optional boolean mask of valid pixels to keep.
    :param use_canny: (bool) If `True`, use `Canny` edge detection, otherwise `Sobel`.
    :return: (ndarray) (h, w) Detected depth edges in the image.
    """
    if preprocess not in {'log', 'inv', 'none', None}:
        raise ValueError(f'Invalid depth preprocessing. ({preprocess})')

    depth = depth.squeeze()
    if preprocess == 'log':
        depth = to_log(depth)
    elif preprocess == 'inv':
        depth = to_inv(depth)
        depth -= depth.min()
        depth /= depth.max()
    else:
        depth = torch.tensor(depth)
        input_value = (depth > 0) * depth.clamp(min=eps(depth))
        # depth = torch.log(input_value) / torch.log(torch.tensor(1.9))
        # depth = torch.log(input_value) / torch.log(torch.tensor(1.9))
        depth = torch.log(input_value) / torch.log(torch.tensor(1.5))
        
    depth = depth.numpy()

    if use_canny:
        edges = canny(depth, sigma=sigma, mask=mask)
    else:
        raise NotImplementedError("Sobel edge detection is not implemented yet.")

    return edges

def EdgeAcc(
    pred, 
    gt, 
    valid_mask,
    th_edges_acc=10,
    th_edges_comp=10,):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """

    pred_edges = extract_edges(pred.detach().cpu(), use_canny=True, preprocess='log')
    pred_edges = pred_edges > 0
    pred_edges = pred_edges.squeeze()
    
    gt_edges = extract_edges(gt.detach().cpu(), use_canny=True, preprocess='log')
    gt_edges = gt_edges > 0
    gt_edges = gt_edges.squeeze()
        
    valid_mask = valid_mask.detach().cpu().numpy()
    invalid_mask = np.logical_not(valid_mask)
    
    # D_target = ndimage.distance_transform_edt(1 - gt_edge_update)
    D_target = ndimage.distance_transform_edt(np.logical_not(gt_edges))
    
    # D_pred = ndimage.distance_transform_edt(1 - pred_edges)  # Distance of each pixel to predicted edges
    D_pred = ndimage.distance_transform_edt(np.logical_not(pred_edges))  # Distance of each pixel to predicted edges
    
    gt_edges[invalid_mask] = 0
    pred_edges[invalid_mask] = 0
    
    pred_edges_BDE = pred_edges & (D_target < th_edges_acc)  # Predicted edges close enough to real ones. (This is from the offical repo)
    gt_edge_BDE = gt_edges & (D_pred < th_edges_comp)  # Real edges close enough to predicted ones.
    
    metric = {
        'EdgeAcc': D_target[pred_edges_BDE].mean() if pred_edges_BDE.sum() else th_edges_acc,  # Distance from pred to target
        'EdgeComp': D_pred[gt_edges].mean() if pred_edges_BDE.sum() else th_edges_comp,  # Distance from target to pred
    }
    
    metric['EdgeAcc'] = torch.tensor(metric['EdgeAcc'])
    return metric['EdgeAcc']

def EdgeComp(
    pred, 
    gt, 
    valid_mask,
    th_edges_acc=10,
    th_edges_comp=10,):
    
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """

    pred_edges = extract_edges(pred.detach().cpu(), use_canny=True, preprocess='log')
    pred_edges = pred_edges > 0
    pred_edges = pred_edges.squeeze()
    
    gt_edges = extract_edges(gt.detach().cpu(), use_canny=True, preprocess='log')
    gt_edges = gt_edges > 0
    gt_edges = gt_edges.squeeze()
    
    valid_mask = valid_mask.detach().cpu().numpy()
    invalid_mask = np.logical_not(valid_mask)
    
    # D_target = ndimage.distance_transform_edt(1 - gt_edge_update)
    D_target = ndimage.distance_transform_edt(np.logical_not(gt_edges))
    
    # D_pred = ndimage.distance_transform_edt(1 - pred_edges)  # Distance of each pixel to predicted edges
    D_pred = ndimage.distance_transform_edt(np.logical_not(pred_edges))  # Distance of each pixel to predicted edges
    
    gt_edges[invalid_mask] = 0
    pred_edges[invalid_mask] = 0
    
    pred_edges_BDE = pred_edges & (D_target < th_edges_acc)  # Predicted edges close enough to real ones. (This is from the offical repo)
    gt_edge_BDE = gt_edges & (D_pred < th_edges_comp)  # Real edges close enough to predicted ones.
    
    metric = {
        'EdgeAcc': D_target[pred_edges_BDE].mean() if pred_edges_BDE.sum() else th_edges_acc,  # Distance from pred to target
        'EdgeComp': D_pred[gt_edges].mean() if pred_edges_BDE.sum() else th_edges_comp,  # Distance from target to pred
    }
    
    metric['EdgeComp'] = torch.tensor(metric['EdgeComp'])
    return metric['EdgeComp']


def shift_2d_replace(data, dx, dy, constant=False):
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data

def soft_edge_error(pred, gt, valid_mask, radius=1):
    pred = pred.squeeze().cpu().numpy()
    gt = gt.squeeze().cpu().numpy()
    abs_diff=[]
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            abs_diff.append(np.abs(shift_2d_replace(gt, i, j, 0) - pred))
    see_depth_map = np.minimum.reduce(abs_diff)
    see_depth_map = torch.tensor(see_depth_map)
    see_depth_map_valid = see_depth_map[valid_mask.detach().cpu()]
    see_depth = see_depth_map_valid.mean()
    return see_depth
