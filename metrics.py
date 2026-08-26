import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, probs, targets):
        return self.bce(probs, targets) + self.dice(probs, targets)

def calculate_metrics(probs, targets, threshold=0.5, smooth=1e-6):
    preds = (probs > threshold).float().view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    total_preds = preds.sum()
    total_targets = targets.sum()
    
    union = total_preds + total_targets - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    precision = (intersection + smooth) / (total_preds + smooth)
    recall = (intersection + smooth) / (total_targets + smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    
    return iou.item(), precision.item(), recall.item(), f1.item()
