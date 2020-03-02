from torchvision import models


MODELS = {
    name: models.__dict__[name] for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])}
