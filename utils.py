import torch

def compute_iou(pred, mask):
    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum() - intersection
    return (intersection / (union + 1e-6)).item()

def compute_dice(pred, mask):
    intersection = (pred * mask).sum()
    return (2 * intersection / (pred.sum() + mask.sum() + 1e-6)).item()
