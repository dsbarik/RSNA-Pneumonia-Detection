import torch
import config
from train.checkpoint_manager import CheckpointManager
from models.build import build_model
from torchinfo import summary


def get_device():
    if config.DEVICE == 'mps' and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    

def main():
    
    device = get_device()
    
    ckpt_manager = CheckpointManager(
        config,
        config.CHECKPOINT_METRIC,
        device,
        config.CHECKPOINT_MODE
    )

    model = build_model(
        model_name=config.MODEL_NAME, num_classes=config.NUM_CLASSES
    )
    
    _ = ckpt_manager.load_best(model)
    
    print("Model Architecture:\n")
    summary(model, input_size=(1, 1, 224, 224))
    print("\n")
    
    

if __name__ == '__main__':
    main()