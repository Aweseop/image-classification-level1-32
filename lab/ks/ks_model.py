from typing import Any
import torch
import torch.nn as nn
import torchvision
import numpy as np

# Custom Model Template
class KSResnetModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.net = torchvision.models.resnet50(pretrained=True)
        self.net.fc = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)
        nn.init.xavier_uniform_(self.net.fc.weight)
        stdv = 1/np.sqrt(2048)
        self.net.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.net(x)
