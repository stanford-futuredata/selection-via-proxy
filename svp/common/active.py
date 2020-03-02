import os
from typing import Tuple, Optional, Dict, Any
# from typing import Protocol  # Python 3.8 and above
from typing_extensions import Protocol
from collections import OrderedDict

import torch
from torch import nn
from torch.optim import Optimizer  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from svp.common import utils
from svp.common.train import run_training
from svp.common.datasets import DatasetWithIndex
from svp.common.selection import UNCERTAINTY_METHODS


SELECTION_METHODS = [
    'kcenters',
    'random'
]
SELECTION_METHODS += UNCERTAINTY_METHODS


class CreateGraph(Protocol):
    def __call__(self, run_dir: Optional[str]) -> Tuple[nn.Module, Optimizer]:
        pass


def validate_splits(train_dataset: Dataset, validation: int,
                    initial_subset: int, rounds: Tuple[int, ...]):
    """
    Ensure there is enough data for validation and selection rounds.
    """
    num_train = len(train_dataset)
    assert validation < num_train and validation >= 0

    num_train -= validation
    assert initial_subset <= num_train and initial_subset > 0

    num_train -= initial_subset
    assert all(round_ > 0 for round_ in rounds)
    assert sum(rounds) <= num_train


def check_different_models(config: Dict[str, Any]) -> bool:
    for key in config.keys():
        if key.startswith('proxy_') and 'eval' not in key:
            if config[key] != config[key.replace('proxy_', '')]:
                return True
    return False


def generate_models(create_graph: CreateGraph,
                    epochs: Tuple[int, ...], learning_rates: Tuple[float, ...],
                    train_dataset: Dataset, batch_size: int,
                    device: torch.device, use_cuda: bool,
                    num_workers: int = 0,
                    device_ids: Optional[Tuple[int, ...]] = None,
                    dev_loader: Optional[DataLoader] = None,
                    test_loader: Optional[DataLoader] = None,
                    fp16: bool = False,
                    loss_scale: float = 256.0,
                    criterion: Optional[nn.Module] = None,
                    run_dir: Optional[str] = None, checkpoint: str = 'last'):
    model = None
    round_dir = None
    stats: Dict[str, Any] = OrderedDict()
    while True:
        labeled = yield model, stats
        stats.clear()

        if run_dir is not None:
            round_dir = os.path.join(run_dir, str(len(labeled)))
            os.makedirs(round_dir, exist_ok=True)
            utils.save_index(labeled, round_dir,
                             'labeled_{}.index'.format(len(labeled)))

        # Create training loader for labelled indices
        train_sampler = SubsetRandomSampler(labeled)
        train_loader = torch.utils.data.DataLoader(
            DatasetWithIndex(train_dataset), sampler=train_sampler,
            batch_size=batch_size, num_workers=num_workers,
            pin_memory=use_cuda)

        # Create the model and optimizer for training.
        model, optimizer = create_graph(run_dir=round_dir)
        # Move the model to the appropriate device.
        model = model.to(device)
        # Create the loss criterion and move it to the device.
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        if fp16:
            from apex import amp  # avoid dependency unless necessary.
            model, optimizer = amp.initialize(model, optimizer,
                                              loss_scale=loss_scale)
        if use_cuda:
            assert model is not None  # mypy hack
            model = nn.DataParallel(model, device_ids=device_ids)

        # Run Training
        assert model is not None  # mypy hack
        model, accuracies, times = run_training(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train_loader=train_loader,
            epochs=epochs,
            learning_rates=learning_rates,
            dev_loader=dev_loader,
            test_loader=test_loader,
            fp16=fp16,
            run_dir=round_dir,
            checkpoint=checkpoint)

        # Save details for easy access and analysis.
        stats['nexamples'] = len(labeled)
        stats['train_accuracy'] = accuracies.train
        stats['dev_accuracy'] = accuracies.dev
        stats['test_accuracy'] = accuracies.test

        stats['train_time'] = times.train
        stats['dev_time'] = times.dev
        stats['test_time'] = times.test


def symlink_target_to_proxy(run_dir: str):
    """
    Create symbolic links from proxy files to the corresponding target files.

    Parameters
    ----------
    run_dir : str
    """
    # Symlink target directory and files to proxy becasue they
    #   are the same.
    print('Proxy and target are not different.')
    proxy_dir = os.path.join(run_dir, 'proxy')
    target_dir = os.path.join(run_dir, 'target')
    os.symlink(os.path.relpath(proxy_dir, run_dir), target_dir)
    print(f'Linked {target_dir} to {proxy_dir}')

    proxy_csv = os.path.join(run_dir, 'proxy.csv')
    target_csv = os.path.join(run_dir, 'target.csv')
    os.symlink(os.path.relpath(proxy_dir, run_dir), target_csv)
    print(f'Linked {target_csv} to {proxy_csv}')


def symlink_to_precomputed_proxy(previous_dir: str, current_dir: str):
    """
    Create symbolic links from previously computed proxy to current run.

    Parameters
    ----------
    precomputed_selection : str
    run_dir : str
    """
    proxy_csv = os.path.join(previous_dir, 'proxy.csv')
    os.symlink(os.path.relpath(proxy_csv, current_dir),
               os.path.join(current_dir, 'proxy.csv'))

    selection_csv = os.path.join(previous_dir, 'selection.csv')
    os.symlink(os.path.relpath(selection_csv, current_dir),
               os.path.join(current_dir, 'selection.csv'))

    selection_dir = os.path.join(previous_dir, 'proxy')
    os.symlink(os.path.relpath(selection_dir, current_dir),
               os.path.join(current_dir, 'proxy'))
