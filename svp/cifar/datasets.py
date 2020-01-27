import os
from typing import Optional, List, Callable, Any

from torch.utils.data import Dataset
from torchvision import datasets, transforms

Transform = Callable[[Any], Any]

DATASETS = [
    'cifar10', 'cifar100',
]

MEANS = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4866, 0.4409),
}

STDS = {
    'cifar10': (0.24703223, 0.24348512, 0.26158784),
    'cifar100': (0.26733428, 0.25643846, 0.27615047),
}


def _create_test_dataset(dataset, dataset_dir, transform,
                         target_transform=None):
    if dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root=dataset_dir, train=False,
                                        download=True,
                                        transform=transform,
                                        target_transform=target_transform)
    elif dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(root=dataset_dir, train=False,
                                         download=True,
                                         transform=transform,
                                         target_transform=target_transform)
    else:
        raise NotImplementedError(f'{dataset} is not an available option.')
    return test_dataset


def _create_train_dataset(dataset, dataset_dir, transform,
                          target_transform=None):
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=dataset_dir, train=True,
                                         download=True,
                                         transform=transform,
                                         target_transform=target_transform)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=dataset_dir, train=True,
                                          download=True,
                                          transform=transform,
                                          target_transform=target_transform)
    else:
        raise NotImplementedError(f'{dataset} is not an available option.')

    return train_dataset


def create_dataset(dataset: str, datasets_dir: str,
                   transform: Optional[List[Transform]] = None,
                   target_transform: Optional[List[Transform]] = None,
                   train: bool = True,
                   augmentation: bool = True) -> Dataset:
    """
    Create CIFAR datasets.

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
        raw_transforms = transform
    else:
        raw_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(MEANS[dataset], STDS[dataset])]

    if augmentation:
        if not train:
            print("Warning: using augmentation on eval data")
        raw_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ] + raw_transforms
    _transform = transforms.Compose(raw_transforms)

    if train:
        _dataset = _create_train_dataset(dataset, dataset_dir,
                                         transform=_transform,
                                         target_transform=target_transform)
    else:
        _dataset = _create_test_dataset(dataset, dataset_dir,
                                        transform=_transform,
                                        target_transform=target_transform)
    print(_dataset)
    return _dataset
