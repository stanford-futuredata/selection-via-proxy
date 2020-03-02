from datetime import datetime
from collections import OrderedDict
from typing import Tuple, Optional, Dict, Any

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from svp.common import utils
from svp.common.selection.k_center_greedy import k_center_greedy


UNCERTAINTY_METHODS = [
    'least_confidence',
    'entropy',
]
ALL_SELECTION_METHODS = [
    'kcenters',
    'random',
    'forgetting_events'
]
ALL_SELECTION_METHODS += UNCERTAINTY_METHODS


def select(model: nn.Module, dataset: Dataset, current: np.array,
           pool: np.array, budget: int, method: str, batch_size: int,
           device: torch.device, device_ids: Tuple[int, ...], num_workers: int,
           use_cuda: bool, keep: str = 'fc',
           nevents: Optional[np.array] = None) -> Tuple[np.array, Dict]:
    '''
    Select a subset of examples from a Dataset.

    Parameters
    ----------
    model : nn.Module
    dataset : Dataset
    current : np.array
    pool : np.array
    budget : int
    method : str
    batch_size : int
    device : torch.device
    device_ids : Tuple[int]
    num_workers : int
    use_cuda : bool
    keep : str, name of final layer of model
    nevents : np.array

    Returns
    -------
    update : numpp.array
        New subset containing both new and `current` indices.
    stats : Dict
        Metadata about how long selection took.
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
    elif method in UNCERTAINTY_METHODS:
        current_set = set(current)
        candidates = np.array([i for i in pool if i not in current_set])
        preds, _ = _calc_preds_and_features(
            model, dataset, candidates, batch_size, num_workers, device,
            device_ids, use_cuda)

        _ranking_start = datetime.now()
        _inference_time = _ranking_start - _inference_start

        if method == 'least_confidence':
            probs = preds.max(axis=1)
            indices = probs.argsort(axis=0)
        elif method in 'entropy':
            entropy = (np.log(preds) * preds).sum(axis=1) * -1.
            indices = entropy.argsort(axis=0)[::-1]
        else:
            raise NotImplementedError(f"'{method}' method doesn't exist")
        ranked = candidates[indices]  # Map back to original indices
        new = ranked[:budget]
    elif method == 'kcenters':
        candidates = np.array(list((set(pool) - set(current))))

        # Make it easy to map back to indices and specify current subset
        subset_indices = np.concatenate([current, candidates])
        _, features = _calc_preds_and_features(
            model, dataset, subset_indices, batch_size, num_workers, device,
            device_ids, use_cuda, keep=keep)

        _ranking_start = datetime.now()
        _inference_time = _ranking_start - _inference_start

        new = k_center_greedy(features, np.arange(len(current)), budget)
        assert (new >= len(current)).all()
        new = subset_indices[new]
    elif method == 'forgetting_events':
        assert nevents is not None
        _ranking_start = datetime.now()
        _inference_time = _ranking_start - _inference_start
        ranked = nevents.argsort()[::-1]
        curr_set = set(current)
        pool_set = set(pool)
        ranked = [index for index in ranked if index in pool_set]
        ranked = [index for index in ranked if index not in curr_set]
        new = ranked[:budget]
    else:
        raise NotImplementedError(f"'{method}' method doesn't exist")

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


def _calc_preds_and_features(
        model: nn.Module, dataset: Dataset, subset: np.array, batch_size: int,
        num_workers: int, device: torch.device, device_ids: Tuple[int, ...],
        use_cuda: bool, keep: Optional[str] = None):
    """
    Calculate predictions and features for selection methods.
    """
    subset_dataset = Subset(dataset, subset)
    loader = torch.utils.data.DataLoader(  # type: ignore
        subset_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda)

    model.eval()
    if keep is not None:
        # TODO: fix this for data parallel training
        # model = model.module
        # model.to(device)
        model = utils.RecordInputs(model, keep=[keep])

    _features = []
    _preds = []
    with torch.no_grad():
        wrapped_loader = tqdm(loader, desc="Inference on unlabeled pool")
        for index, (inputs, targets) in enumerate(wrapped_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            dist = torch.nn.functional.softmax(outputs, dim=1)
            _preds.append(dist.detach().cpu())

            if keep is not None:
                _features.append(model.kept[keep].cpu())  # type: ignore

    preds = torch.cat(_preds).numpy()
    if keep is not None:
        features = torch.cat(_features).numpy()
    else:
        features = None

    return preds, features
