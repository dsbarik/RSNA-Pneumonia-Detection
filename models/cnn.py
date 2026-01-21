import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Lightweight CNN for chest X-ray classification.
    Designed to be trained from scratch.
    """

    def __init__(self, num_outputs: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
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
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(256, num_outputs)

    def forward(self, x):
        """
        Input:  (B, 1, 224, 224)
        Output: (B, num_outputs)
        """
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
