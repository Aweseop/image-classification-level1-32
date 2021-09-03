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
class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.resnet50.fc.weight)
        stdv =  1 / math.sqrt(self.resnet50.fc.in_features)
        self.resnet50.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.resnet50(x)
        return x

# Custom Model Template
class EfficientNet_b0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.model = timm.create_model('tf_efficientnet_b0', pretrained = True)
        self.model.classifier = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv =  1 / math.sqrt(self.model.classifier.in_features)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

# Custom Model Template
class EfficientNet_b1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.model = timm.create_model('tf_efficientnet_b1', pretrained = True)
        self.model.classifier = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv =  1 / math.sqrt(self.model.classifier.in_features)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

# Custom Model Template
class EfficientNet_b2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.model = timm.create_model('tf_efficientnet_b2', pretrained = True)
        self.model.classifier = nn.Linear(in_features=1408, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv =  1 / math.sqrt(self.model.classifier.in_features)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

# Custom Model Template
class EfficientNet_b3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.model = timm.create_model('tf_efficientnet_b3', pretrained = True)
        self.model.classifier = nn.Linear(in_features=1536, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv =  1 / math.sqrt(self.model.classifier.in_features)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

# Custom Model Template
class EfficientNet_b4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.model = timm.create_model('tf_efficientnet_b4', pretrained = True)
        self.model.classifier = nn.Linear(in_features=1792, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv =  1 / math.sqrt(self.model.classifier.in_features)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

# Custom Model Template
class EfficientNet_b5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.model = timm.create_model('tf_efficientnet_b5', pretrained = True)
        self.model.classifier = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv =  1 / math.sqrt(self.model.classifier.in_features)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

# Custom Model Template
class Vit_large(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
        self.model.head = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.head.weight)
        stdv =  1 / math.sqrt(self.model.head.in_features)
        self.model.head.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

# Custom Model Template
class Vit_base(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.head.weight)
        stdv =  1 / math.sqrt(self.model.head.in_features)
        self.model.head.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

# Custom Model Template
class Vit_base_resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.model = timm.create_model('vit_base_resnet50_384', pretrained=True)
        self.model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.head.weight)
        stdv =  1 / math.sqrt(self.model.head.in_features)
        self.model.head.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

# Custom Model Template
class efficientnet_b1_pruned(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.model = timm.create_model('efficientnet_b1_pruned', pretrained=True)
        self.model.classifier = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv =  1 / math.sqrt(self.model.classifier.in_features)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

# Custom Model Template
class efficientnet_b2_pruned(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.model = timm.create_model('efficientnet_b2_pruned', pretrained=True)
        self.model.classifier = nn.Linear(in_features=1408, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv =  1 / math.sqrt(self.model.classifier.in_features)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

# Custom Model Template
class efficientnet_b3_pruned(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.model = timm.create_model('efficientnet_b3_pruned', pretrained=True)
        self.model.classifier = nn.Linear(in_features=1536, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv =  1 / math.sqrt(self.model.classifier.in_features)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

# Custom Model Template
class Resnet50_for_xgboost(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        # self.resnet50.fc = Identity()

        
    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        x = self.resnet50.avgpool(x)
        
        return x

from xgboost import XGBClassifier

xgbmodel = XGBClassifier(objective='multi:softprob', num_class= 18)

from catboost import CatBoostClassifier, Pool

model_cat = CatBoostClassifier()