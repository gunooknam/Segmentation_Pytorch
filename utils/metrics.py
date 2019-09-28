import numpy as np

def accuracy(seg, pred):
    total = np.sum(seg > -1)
    return np.sum( seg==pred ) / total

def iou(seg, pred):
    unique_classes = [c for c in np.unique(seg) if c!= -1]
    iou = 0.0
    for c in unique_classes:
        pred_c = pred == c
        seg_c = seg == c
        intersection = np.logical_and(seg_c, pred_c)
        union = np.logical_or(seg_c, pred_c)
        iou += np.sum(intersection) / (np.sum(union) + 1e-6)
    return iou / len(unique_classes)