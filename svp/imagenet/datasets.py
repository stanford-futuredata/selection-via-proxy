import os
from typing import Optional, List, Callable, Any

from torch.utils.data import Dataset
from torchvision import datasets, transforms


# Obivous, but adding this to be consistent with other datasets.
DATASETS = ['imagenet']


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


Transform = Callable[[Any], Any]


def create_dataset(dataset: str, datasets_dir: str,
                   transform: Optional[List[Transform]] = None,
                   target_transform: Optional[List[Transform]] = None,
                   train: bool = True,
                   augmentation: bool = True) -> Dataset:
    """
    Create ImageNet dataset.

    Parameters
    ----------
    dataset: str
        Name of dataset.
    datasets_dir: str
        Base directory for datasets
    transform: list of transforms or None, default None
        Transform the inputs.
    target_transform: list of transforms or None, default None
        Transform the outputs.
    train: bool, default True
        Load training data.
    augmentation: bool, default True
        Apply default data augmentation for training.

    Returns
    -------
    _dataset: Dataset
    """
    dataset_dir = os.path.join(datasets_dir, dataset)
    if transform is not None:
        base_transforms = transform
    else:
        base_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)]

    if train:
        train_dir = os.path.join(dataset_dir, 'train')
        transform_train: List[Transform] = []
        if augmentation:
            transform_train = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip()
            ]
        transform_train = transforms.Compose(transform_train + base_transforms)
        # TODO: Could update this to dataset.ImageNet.
        _dataset = datasets.ImageFolder(train_dir,
                                        transform=transform_train,
                                        target_transform=target_transform)

    else:
        val_dir = os.path.join(dataset_dir, 'val')
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ] + base_transforms)
        # TODO: Could update this to dataset.ImageNet.
        _dataset = datasets.ImageFolder(val_dir,
                                        transform=transform_val,
                                        target_transform=target_transform)

    print(_dataset)
    return _dataset
