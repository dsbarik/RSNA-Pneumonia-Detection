from models.cnn import BaselineCNN
from models.cnn_v2 import CNNV2
from models.hybrid_cnn import HybridCNN
from models.resnet_18 import Resnet18
from models.resnet_50 import Resnet50
from models.resnet_101 import Resnet101
from models.small_resnet import SmallResNet

MODEL_REGISTRY = {
    "baseline": BaselineCNN,
    "small_resnet": SmallResNet,
    "hybrid_cnn": HybridCNN,
    "cnn_v2": CNNV2,
    "resnet18": Resnet18,
    "resnet50": Resnet50,
    "resnet101": Resnet101
}