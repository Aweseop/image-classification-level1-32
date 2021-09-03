from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import math

class JHSample():
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print('hello im jh')

# Custom Model Template
class EfficientNet_b0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.efficientnet_b0 = timm.create_model('tf_efficientnet_b0', pretrained = True)
        self.efficientnet_b0.classifier = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.efficientnet_b0.classifier.weight)
        stdv =  1 / math.sqrt(self.efficientnet_b0.classifier.in_features)
        self.efficientnet_b0.classifier.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.efficientnet_b0(x)
        return x

class EfficientNet_b3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.efficientnet = timm.create_model('tf_efficientnet_b3', pretrained = True)
        self.efficientnet.classifier = nn.Linear(in_features=1792, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.efficientnet.classifier.weight)
        stdv =  1 / math.sqrt(self.efficientnet.classifier.in_features)
        self.efficientnet.classifier.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.efficientnet(x)
        return x