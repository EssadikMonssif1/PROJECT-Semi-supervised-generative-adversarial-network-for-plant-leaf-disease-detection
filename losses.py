import torch.nn as nn

bce = nn.BCELoss()
ce = nn.BCELoss()

def generator_loss(pred_mask, real_mask):
    return ce(pred_mask, real_mask)

def discriminator_loss(pred, target):
    return bce(pred, target)
