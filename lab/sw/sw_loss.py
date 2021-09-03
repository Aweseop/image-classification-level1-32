from typing import Any
from loss import FocalLoss, F1Loss
import torch.nn as nn

class SWLoss():
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print('hello im sw')

class FocalF1Loss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.focal = FocalLoss()
        self.f1 = F1Loss()
    
    def forward(self,input_tensor, target_tensor):
        focal = self.focal(input_tensor, target_tensor)
        f1 = self.f1(input_tensor, target_tensor)

        return f1+focal

sw_criterion_entrypoints = {
    'swloss': SWLoss,
    'ff': FocalF1Loss
}
