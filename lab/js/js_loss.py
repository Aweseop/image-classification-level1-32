from typing import Any
import torch.nn as nn
import torch
import torch.nn.functional as F

class JSLoss():
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print('hello im js')

class custom_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true, model_param):
        y_true = F.one_hot(y_true, num_classes=18)
        regularization_loss = torch.mean(torch.square(model_param))
        hinge_loss = torch.mean(
            torch.square(
                torch.maximum(
                    torch.zeros([y_true.size()[0], 18]).to('cuda'), 1 - y_pred * y_true
                )
            )
        )
        loss = regularization_loss * 1000 + 0.15 * hinge_loss
        return loss


js_criterion_entrypoints = {
    'custom_loss': custom_loss
}