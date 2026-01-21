import torch
import torch.nn as nn


class FixedStem(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- LAYER 1: Gaussian Blur (Fixed) ---
        # Reduces grain/noise so the edge detectors work better
        self.blur = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        gaussian_k = torch.tensor([[1., 2., 1.],
                                   [2., 4., 2.],
                                   [1., 2., 1.]]) / 16.0
        self.blur.weight.data = gaussian_k.view(1, 1, 3, 3)
        self.blur.weight.requires_grad = False  # <--- FREEZE

        # --- LAYER 2: Filter Bank (Fixed) ---
        # Input: 1 channel (blurred image)
        # Output: 4 channels (Blob, SobelX, SobelY, Identity)
        self.bank = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        
        # Define the 4 filters
        laplacian = torch.tensor([[0., -1., 0.], [-1., 8., -1.], [0., -1., 0.]])
        sobel_x   = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y   = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        identity  = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        
        # Stack them into shape (4, 1, 3, 3)
        filters = torch.stack([laplacian, sobel_x, sobel_y, identity], dim=0) # Shape: (4, 3, 3)
        filters = filters.unsqueeze(1)  # Shape: (4, 1, 3, 3)
        
        self.bank.weight.data = filters
        self.bank.weight.requires_grad = False # <--- FREEZE

    def forward(self, x):
        x = self.blur(x)
        x = self.bank(x)
        return x

class HybridCNN(nn.Module):
    def __init__(self, num_outputs=2):
        super().__init__()
        
        # 1. The Fixed Front-End
        self.stem = FixedStem()
        
        # 2. The Learnable Network
        # NOTE: in_channels=4 because our FixedStem outputs 4 feature maps
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_outputs)

    def forward(self, x):
        # Step 1: Extract fixed features (Blur -> Edges/Blobs)
        x = self.stem(x) 
        
        # Step 2: Learn to mix them
        x = self.features(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x