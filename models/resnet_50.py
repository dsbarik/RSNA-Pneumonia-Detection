import torch.nn as nn
from torchvision import models


class Resnet50(nn.Module):
    def __init__(self, num_outputs: int=2):
        super().__init__()
        
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        for param in self.resnet50.parameters():
            param.requires_grad = False
        
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_outputs)
        
    def forward(self, x):
        x = self.resnet50(x)
        return x