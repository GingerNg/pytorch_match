import torch.nn as nn
from losses.focal_loss import FocalLoss
from losses.contrastive_loss import ContrastiveLoss


def cross_entropy_loss():
    criterion = nn.CrossEntropyLoss()  # obj
    return criterion


def focal_loss():
    criterion = FocalLoss(4)  # obj
    return criterion


def binarg_loss():
    # return nn.BCEWithLogitsLoss()
    return nn.BCELoss()


def contrastive_loss():
    return ContrastiveLoss()
