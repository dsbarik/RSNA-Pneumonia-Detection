import torch.nn as nn
from torchvision import models


class Resnet18(nn.Module):
    def __init__(self, num_outputs: int=2):
        super().__init__()
        
        self.resnet18 = models.resnet18(weights=None)
        
        # for param in self.resnet18.parameters():
            # param.requires_grad = False
        
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, num_outputs)
        
    def forward(self, x):
        x = self.resnet18(x)
        return x