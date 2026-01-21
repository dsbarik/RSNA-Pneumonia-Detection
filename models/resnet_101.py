from torch import nn
from torchvision import models


class Resnet101(nn.Module):
    def __init__(self, num_outputs: int=2):
        super().__init__()
        
        self.resnet101 = models.resnet101(models.ResNet101_Weights.DEFAULT)
        
        for param in self.resnet101.parameters():
            param.requires_grad = False
        
        num_features = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Linear(num_features, num_outputs)
        
    def forward(self, x):
        x = self.resnet101(x)
        return x
        