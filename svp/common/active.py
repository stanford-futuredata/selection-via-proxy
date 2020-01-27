import os
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, Callable
from collections import OrderedDict

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.optim import Optimizer  # type: ignore
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler

from svp.common import utils
from svp.common.train import run_training
from svp.common.selection.k_center_greedy import k_center_greedy


PROXY_DIR_PREFIX = 'selector'
TARGET_DIR_PREFIX = 'target'
SELECTION_METHODS = ['greedy', 'least_confidence', 'random']


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


def create_eval_loaders(train_dataset, valid_dataset, test_dataset,
                        validation: int, shuffle: bool, run_dir: str,
                        target_eval_batch_size: int,
                        proxy_eval_batch_size: int,
                        num_workers: int, use_cuda: bool):
    train_indices, valid_indices = utils.split_indices(
        train_dataset, validation, run_dir, shuffle=shuffle)
    valid_sampler = SubsetRandomSampler(valid_indices)
    proxy_test_loader = DataLoader(
        test_dataset, batch_size=proxy_eval_batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
    if len(valid_indices) > 0:
        proxy_valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=proxy_eval_batch_size,
            sampler=valid_sampler, num_workers=num_workers,
            pin_memory=use_cuda)
    else:
        print('Using test dataset for validation')
        proxy_valid_loader = proxy_test_loader

    # Only create target loaders if they will be used
    if target_eval_batch_size != proxy_eval_batch_size:
        target_test_loader = DataLoader(
            test_dataset, batch_size=target_eval_batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
        if len(valid_indices) > 0:
            target_valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=target_eval_batch_size,
                sampler=valid_sampler, num_workers=num_workers,
                pin_memory=use_cuda)
        else:
            target_valid_loader = target_test_loader
    else:
        target_valid_loader = proxy_valid_loader
        target_test_loader = proxy_test_loader

    return (train_indices,
            proxy_valid_loader, proxy_test_loader,
            target_valid_loader, target_test_loader)


def check_different_models(config: Dict[str, Any]) -> bool:
    for key in config.keys():
        if key.startswith('proxy_') and 'eval' not in key:
            if config[key] != config[key.replace('proxy_', '')]:
                return True
    return False


def create_trainer(run_dir: str, num_classes: int, train_dataset: Dataset,
                   valid_loader: DataLoader, test_loader: DataLoader,
                   create_graph: Callable[[str], Tuple[nn.Module, Optimizer]],
                   epochs: Tuple[int, ...], learning_rates: Tuple[float, ...],
                   batch_size: int, num_workers: int,
                   device: torch.device, device_ids: Tuple[int, ...],
                   checkpoint: str, use_cuda: bool, track_test_acc: bool,
                   prefix: Optional[str] = None):
    model = None
    round_dir = None
    stats: Dict[str, Any] = OrderedDict()
    while True:
        labelled = yield model, stats
        stats.clear()

        round_dir = os.path.join(run_dir, prefix, str(len(labelled)))  # type: ignore  # noqa: E501
        os.makedirs(round_dir, exist_ok=True)
        utils.save_index(labelled, round_dir,
                         'labelled_{}.index'.format(len(labelled)))

        # Create training loader for labelled indices
        train_sampler = SubsetRandomSampler(labelled)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=train_sampler, batch_size=batch_size,
            num_workers=num_workers, pin_memory=use_cuda)

        model, optimizer = create_graph(round_dir)

        criterion = nn.CrossEntropyLoss()
        model = model.to(device)
        if use_cuda:
            model = nn.DataParallel(model, device_ids=device_ids)
        criterion = criterion.to(device)

        model, accuracy, train_time, eval_time = run_training(
            run_dir=round_dir,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            epochs=epochs,
            learning_rates=learning_rates,
            checkpoint=checkpoint,
            use_cuda=use_cuda,
            track_test_acc=track_test_acc)

        stats[f"nexamples"] = len(labelled)
        stats[f"accuracy"] = accuracy
        stats[f"train_time"] = train_time
        stats[f"eval_time"] = eval_time


def select(model: nn.Module, dataset: Dataset, current: np.array,
           pool: np.array, budget: int, method: str, batch_size: int,
           device: torch.device, device_ids: Tuple[int, ...], num_workers: int,
           use_cuda: bool, keep: str = 'fc'):
    '''
    Args
    - model: torch.nn.Module, already on device
    - dataset: torch.utils.data.Dataset
    - current: np.array, shape [num_selected], type int64
    - pool: np.array, shape [N], type int64
    - budget: int, number of points to select
    - method: str, one of SELECTION_METHODS
    - config: dict
    - device: torch.device
    - keep: str, name of final layer of model
    Returns: np.array, shape [num_selected + N*size], type int64 new subset
    '''
    stats: Dict[str, Any] = OrderedDict()
    _total_start = datetime.now()
    assert not model.training
    N = len(pool)
    num_selected = len(current)
    print(f'Selecting {budget} new examples (curr={num_selected}, pool={N})')

    _inference_start = datetime.now()
    if method == 'random':
        _ranking_start = datetime.now()
        _inference_time = _ranking_start - _inference_start
        candidates = list(set(pool) - set(current))
        new = np.random.permutation(candidates)[:budget]
    elif method == 'least_confidence':
        # Least Confidence:
        current_set = set(current)
        candidates = np.array([i for i in pool if i not in current_set])
        subset_dataset = Subset(dataset, candidates)
        loader = torch.utils.data.DataLoader(  # type: ignore
            subset_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_cuda)

        _preds = []
        with torch.no_grad():
            for index, (inputs, targets) in enumerate(tqdm(loader)):
                inputs = inputs.to(device)
                outputs = model(inputs)

                dist = torch.nn.functional.softmax(outputs, dim=1)
                _preds.append(dist.detach().cpu())

        preds = torch.cat(_preds)

        _ranking_start = datetime.now()
        _inference_time = _ranking_start - _inference_start

        probs = preds.max(dim=1)[0]
        # Sort ascending rather than do 1 - probs
        indices = probs.sort(dim=0, descending=False)[1].view(-1).numpy()
        ranked = candidates[indices]  # Map back to original indices

        # Sanity checks that the mapping was correct
        assert len(set(ranked).intersection(current_set)) == 0
        assert len(set(ranked).union(current_set)) == len(pool)

        new = ranked[:budget]
    elif method == 'greedy':
        # TODO: Factor out preds and features into another function
        candidates = np.array(list((set(pool) - set(current))))

        # Make it easy to map back to indices and specify current subset
        subset_indices = np.concatenate([current, candidates])
        subset_dataset = Subset(dataset, subset_indices)
        loader = torch.utils.data.DataLoader(  # type: ignore
            subset_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_cuda)

        # TODO: fix this for data parallel training
        # model = model.module
        # model.to(device)
        model = utils.RecordInputs(model.eval(), keep=[keep])
        features = []
        with torch.no_grad():
            for index, (inputs, targets) in enumerate(tqdm(loader)):
                inputs = inputs.to(device)
                _ = model(inputs)
                features.append(model.kept[keep].cpu())
        features = torch.cat(features).numpy()

        _ranking_start = datetime.now()
        _inference_time = _ranking_start - _inference_start

        new = k_center_greedy(features, np.arange(len(current)), budget)
        assert (new >= len(current)).all()
        new = subset_indices[new]

        # Sanity checks that the mapping was correct
        assert len(set(new).intersection(set(current))) == 0
        assert len(set(new).intersection(set(pool))) == len(new)

    updated = np.concatenate([current, new])
    assert len(set(updated)) == len(current) + len(new)

    _total_end = datetime.now()
    _ranking_time = _total_end - _ranking_start
    _total_time = _total_end - _total_start

    stats['nexamples'] = len(updated)
    stats['current_nexamples'] = len(current)
    stats['new_nexamples'] = len(new)
    stats['total_time'] = _total_time
    stats['inference_time'] = _inference_time
    stats['ranking_time'] = _ranking_time

    print("Selection took {} ({} inference + {} ranking)".format(
        _total_time, _inference_time, _ranking_time))
    return updated, stats
