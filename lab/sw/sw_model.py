from typing import Any

class SWSample():
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print('hello im sw')

import timm
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self,modelName,num_classes):
        super().__init__()
        self.pretrained = timm.create_model(modelName,pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.pretrained(x)

class ResNet50(CustomModel):
    def __init__(self,num_classes):
        super().__init__("resnet50",num_classes)