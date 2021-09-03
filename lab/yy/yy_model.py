import torch
import torchvision

class YYResnet50(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.wide_resnet50_2(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features= 2048, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1/(2048**0.5)
        self.model.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)