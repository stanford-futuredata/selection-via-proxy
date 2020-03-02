import os
import json
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Union, TextIO
from collections.abc import Iterable

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset


def override_option(ctx, param, value):
    """
    Use another click option as the default value.

    If `value` is None, remove the prefix from `param` and take the
    value from that option.

    Parameters
    ----------
    ctx : click.core.Context
        Click context to take value from.
    param : click.core.Option
        Name option to override
    value
        Value to potentially override.

    Returns
    -------
    value
        Final value of option.
    """
    if value is None or (isinstance(value, Iterable) and len(value) == 0):  # type: ignore # noqa: E501
        value = ctx.params['_'.join(param.name.split('_')[1:])]
    return value


def set_random_seed(seed: Optional[int] = None) -> int:
    """
    Set the random seed for numpy and torch.

    Parameters
    ----------
    seed: int or None, default None
        Specify initial random seed.

    Returns
    -------
    seed : int
    """
    seed = np.random.randint(10_000) if seed is None else seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore
    return seed


def capture_config(**kwargs) -> Dict[str, Any]:
    """
    Capture arguments for the current experiment and add timestamps.

    Parameters
    ----------
    kwargs: Dict
        Arguments for the current experiment.

    Returns
    -------
    config : Dict
    """
    config = {k: v for k, v in kwargs.items()}
    config['timestamp'] = "{:.0f}".format(datetime.utcnow().timestamp())
    # The local time is for convenience.
    config['local_timestamp'] = str(datetime.now())
    return config


def create_run_dir(run_dir: str, timestamp: Optional[str] = None) -> str:
    """
    Create a timestamped directory for artifacts from an experiment.

    Parameters
    ----------
    run_dir : str
        Base directory for experiments.
    timestamp : str or None, default None
        Specific timestamp for the run.

    Returns
    -------
    timestamp_dir: str
    """
    if timestamp is None:
        timestamp = "{:.0f}".format(datetime.utcnow().timestamp())
    timestamp_dir = os.path.join(run_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    return timestamp_dir


def config_run_env(cuda: Optional[bool] = None,
                   device_ids: Optional[Tuple[int, ...]] = None,
                   num_workers: Optional[int] = None
                   ) -> Tuple[bool, torch.device, Tuple[int, ...], int]:
    """
    Generate standard data laoding and compute parameters.

    Parameters
    ----------
    cuda : bool or None, default None
        Indicate whether to use cuda and GPUs.
    device_ids : tuple of int or None, default None
        GPU IDs to use.
    num_workers : int or None, default None
        Number of workers to use for data loading.

    Returns
    -------
    use_cuda : bool
    device : Tuple
    _device_ids : Tuple[int]
    num_workers : int
    """
    use_cuda = cuda is not None and cuda and torch.cuda.is_available()
    if use_cuda:
        if device_ids is None or len(device_ids) == 0:
            _device_ids = tuple(range(torch.cuda.device_count()))
        else:
            _device_ids = device_ids

        if num_workers is None:
            num_workers = len(_device_ids)

        device = torch.device('cuda', _device_ids[0])
    else:
        _device_ids = tuple()
        if num_workers is None:
            num_workers = 0
        device = torch.device('cpu')
    return use_cuda, device, _device_ids, num_workers


def split_indices(dataset: Dataset, validation: int,
                  run_dir: Optional[str] = None, shuffle: bool = False
                  ) -> Tuple[List[int], List[int]]:
    """
    Split indices between training and validation.

    Parameters
    ----------
    dataset: Dataset
        Dataset to split between train and val.
    validation int
        Number of examples to use for validattion.
    run_dir: str
        Directory to save indices.
    shuffle: bool, default False
        Indicate whether to shuffle data.

    Returns
    -------
    train_indices: List[Int]
    dev_indices: List[Int]
    """
    num_train = len(dataset)
    indices = np.arange(num_train, dtype=np.int64)
    assert num_train > validation and validation >= 0
    split = num_train - validation

    if shuffle:
        np.random.shuffle(indices)

    train_indices = indices[:split]
    if run_dir is not None:
        save_index(train_indices, run_dir, 'train.index')
    num_train = len(train_indices)
    dev_indices = indices[split:]
    if run_dir is not None:
        save_index(dev_indices, run_dir, 'dev.index')
    num_dev = len(dev_indices)

    print(f'Using {num_train} examples for training pool')
    print(f'Using {num_dev} examples for validation')
    return train_indices, dev_indices


def correct(outputs: torch.Tensor, targets: torch.Tensor,
            top: Tuple[int, ...] = (1, )) -> List[torch.Tensor]:
    """
    Calculate how many examples are correct for multiple top-k values

    Parameters
    ----------
    outputs : torch.Tensor
    target : torch.Tensor
    top : Tuple[int], default (1, )

    Returns
    -------
    tops : List[torch.Tensor]
    """
    _, predictions = outputs.topk(max(top), dim=1, largest=True, sorted=True)
    targets = targets.view(-1, 1).expand_as(predictions)

    corrects = predictions.eq(targets).cpu().int().cumsum(1).sum(0)
    tops = list(map(lambda k: corrects.data[k - 1], top))
    return tops


class RecordInputs(nn.Module):
    def __init__(self, model, keep=None):
        """
        Parameters
        ----------
        model: nn.Module
        keep: List[str]
        """
        super().__init__()
        self.model = model
        self.kept = {}
        if keep is not None:
            for name, module in self.model.named_modules():
                name = name.replace('module.', '')  # For GPUs
                if name in keep:
                    self.kept[name] = None
                    self.keep_inputs(module, name)
        assert len(self.kept) > 0

    def keep_inputs(self, module, name):
        """
        Changes the forward() method of the given module to update
        self.kept[name].

        Parameters
        ----------
        module : nn.Module
        name : str
        """
        func = module.forward

        def _forward(inputs, *args, **kwargs):
            self.kept[name] = inputs.clone()
            return func(inputs, *args, **kwargs)
        module.forward = _forward

    def forward(self, inputs):
        return self.model.forward(inputs)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_result(result: Dict, path: Union[str, TextIO],
                write_heading: bool = False):
    """
    Write partial result from iterative process to file.

    Parameters
    ----------
    result : Dict
        Partial result to save.
    path : str or TextIO
        Output to save result. If str, write_heading is ignored, and
        the heading is written based on whether the file already exists.
    write_heading : bool, default False
        Indicate whether to write the CSV heading.
    """
    if isinstance(path, str):
        write_heading = not os.path.exists(path)
        with open(path, mode='a') as out:
            _save_result_helper(out, result, write_heading)
    else:
        _save_result_helper(path, result, write_heading)


def _save_result_helper(file, result, write_heading):
    """
    Refacored out from save_result to prevent repeated code.
    """
    if write_heading:
        file.write(",".join([str(k) for k, v in result.items()]) + '\n')
    file.write(",".join([str(v) for k, v in result.items()]) + '\n')


def save_config(config, run_dir):
    """
    Save timestamp
    """
    path = os.path.join(run_dir, "config_{}.json".format(config['timestamp']))
    with open(path, 'w') as config_file:
        json.dump(config, config_file)
        config_file.write('\n')


def save_index(indices, directory, filename):
    np.savetxt(os.path.join(directory, filename), indices, fmt='%d')


def get_learning_rate(optimizer):
    for group in optimizer.param_groups:
        if 'lr' in group:
            return group['lr']
