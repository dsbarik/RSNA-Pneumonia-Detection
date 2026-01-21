from models.registry import MODEL_REGISTRY


def build_model(model_name: str, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(**kwargs)
