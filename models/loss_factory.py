import torch.nn as nn
from .focal_loss import FocalLoss

def cross_entropy_loss():
    criterion = nn.CrossEntropyLoss()  # obj
    return criterion

def focal_loss():
    criterion = FocalLoss(4)  # obj
    return criterion

def binarg_loss():
    return nn.BCEWithLogitsLoss()
