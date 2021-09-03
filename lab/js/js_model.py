from typing import Any
import torch.nn as nn
import torchvision.models as models

class JSModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = models.resnet50(pretrained = True)
        self.net.fc = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        return self.net(x)
