import torch
import torch.nn as nn
import torch.nn.functional as F

# cross entropy loss input=logits
class CrossEntropyWithLogits(nn.Module):
    def __init__(self, weight=None,
                 reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, labels):
        log_prob = F.log_softmax(logits, dim=-1)
        return torch.sum(-labels * log_prob, dim=-1).mean()

# focal loss input=logits
class FocalLossWithLogits(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)
        f1 = ((1 - prob) ** self.gamma) * log_prob
        return torch.sum(-labels * f1, dim=-1).mean()

yy_criterion_entrypoints = {
    'yyloss': CrossEntropyWithLogits,
    'yyfocal': FocalLossWithLogits,
    'multi_label_soft_margin': nn.MultiLabelSoftMarginLoss # loss function for multi label classification
}