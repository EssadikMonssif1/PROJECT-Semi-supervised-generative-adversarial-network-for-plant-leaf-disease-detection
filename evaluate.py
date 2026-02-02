import torch

def iou(pred, target):
    pred = pred > 0.5
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return intersection / (union + 1e-6)

