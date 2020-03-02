import os
from datetime import datetime, timedelta
from contextlib import ExitStack
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict, Any, Callable, NamedTuple

import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from svp.common import utils
from svp.common.datasets import DatasetWithIndex

Loaders = Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]


def create_loaders(train_dataset: Dataset,
                   batch_size: int = 128, eval_batch_size: int = 128,
                   validation: int = 0,
                   run_dir: Optional[str] = None,
                   dev_dataset: Optional[Dataset] = None,
                   test_dataset: Optional[Dataset] = None,
                   shuffle: bool = False, use_cuda: bool = False,
                   num_workers: int = 0,
                   eval_num_workers: Optional[int] = None,
                   indices: Optional[Tuple[List[int], List[int]]] = None
                   ) -> Loaders:
    """
    Create data loaders for train, validation, and test.

    Parameters
    ----------
    train_dataset : Dataset
    batch_size : int, default 128
    eval_batch_size : int, defeault 128
    validation : int, default 0
    run_dir : str or None, default None
    dev_dataset : Dataset or None, default None
    test_dataset : Dataset or None, default None
    shuffle : bool, default False
    use_cuda : bool, default False
    num_workers : int, default 0
    eval_num_workers : int or None, default None
    indices : Tuple[List[int], List[int]] or None, default None


    Returns
    -------
    train_loader : DataLoader
    dev_loader : DataLoader or None
    test_loader : DataLoader or None
    """
    # Maybe split the training dataset between training and validation.
    dev_indices: Optional[List[int]] = None
    if indices is None:
        if validation > 0:
            train_indices, dev_indices = utils.split_indices(
                train_dataset, validation, run_dir, shuffle=shuffle)
        else:
            train_indices = np.arange(len(train_dataset), dtype=np.int64)
            dev_indices = None
    else:
        train_indices, dev_indices = indices

    # Create training data loader.
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(
        DatasetWithIndex(train_dataset), sampler=train_sampler,
        batch_size=batch_size, num_workers=num_workers, pin_memory=use_cuda)

    # Use the same number of workers for everything.
    if eval_num_workers is None:
        eval_num_workers = num_workers

    dev_loader = None
    # Create validation data loader.
    if dev_indices is not None and len(dev_indices) > 0:
        # Use part of the training dataset for validation.
        print('Using {} examples from training'
              ' for validation'.format(len(dev_indices)))
        dev_sampler = SubsetRandomSampler(dev_indices)
        dev_loader = torch.utils.data.DataLoader(
            DatasetWithIndex(train_dataset), sampler=dev_sampler,
            batch_size=eval_batch_size, num_workers=eval_num_workers,
            pin_memory=use_cuda)
    elif dev_dataset is not None:
        # Use a separate dataset for valdiation.
        dev_loader = torch.utils.data.DataLoader(
            DatasetWithIndex(dev_dataset), batch_size=eval_batch_size,
            num_workers=eval_num_workers, pin_memory=use_cuda)

    test_loader = None
    # Create test data loader.
    if test_dataset is not None:
        test_loader = DataLoader(
            DatasetWithIndex(test_dataset), batch_size=eval_batch_size,
            shuffle=False, num_workers=eval_num_workers, pin_memory=use_cuda)

    return train_loader, dev_loader, test_loader


class AccuracySplits(NamedTuple):
    train: float
    dev: float
    test: float


class TimeSplits(NamedTuple):
    train: timedelta
    dev: timedelta
    test: timedelta


def run_training(model: nn.Module, optimizer: Optimizer, criterion: nn.Module,
                 device: torch.device, train_loader: DataLoader,
                 epochs: Tuple[int, ...], learning_rates: Tuple[float, ...],
                 dev_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None,
                 batch_callback: Optional[Callable] = None,
                 fp16: bool = False,
                 run_dir: Optional[str] = None,
                 checkpoint: str = 'last'
                 ) -> Tuple[nn.Module, AccuracySplits, TimeSplits]:
    """
    Run training with evaluation after every epoch

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train
    optimizer : Optimizer
        Optimizer to optimize model
    criterion : nn.Module
        Loss function to optimize
    device : torch.device
        Device to place data on
    train_loader : DataLoader
        Training data loader
    epochs : Tuple[int]
        Number of epochs for each learning rate
    learning_rates : Tuple[float]
        Learning rates for optimizer
    dev_loader : DataLoader or None, default None
        Validation data loader (also used for early stopping)
    test_loader : DataLoader or None, default None
        Test data loader
    batch_callback : Callable or None, default None
        Optional function to calculate stats on each batch during training.
    fp16 : bool, default False.
        Use mixed precision training.
    run_dir : str or None, default None
        Directory to save results from training and evaluation
    checkpoint : str, default 'last'
        Type of checkpointing to perform

    Returns
    -------
    model : nn.Module
        Trained model.
    accuracies : Tuple[float, ...]
        The best accuracies from the model on the train, dev, and test splits.
    times : Tuple[timedelta, ...]
        Time spent training or evaluating on the train, dev, and test splits.
    """
    assert len(epochs) == len(learning_rates)
    start_epoch = 1
    global_step = 0

    # Create files for logging results if run_dir is not None.
    train_results_file: Optional[str] = None
    dev_results_file: Optional[str] = None
    test_results_file: Optional[str] = None
    results_file: Optional[str] = None
    if run_dir is not None:
        train_results_file = os.path.join(run_dir, 'train_results.csv')
        if dev_loader is not None:
            dev_results_file = os.path.join(run_dir, 'dev_results.csv')
        if test_loader is not None:
            test_results_file = os.path.join(run_dir, 'test_results.csv')
        results_file = os.path.join(run_dir, 'results.csv')

    # Record model quality on training, validation, and test splits.
    best_accuracy = -1
    best_train_accuracy = -1
    best_dev_accuracy = -1
    best_test_accuracy = -1
    # Record how much time training, validation, and testing take.
    train_time = timedelta(0)
    dev_time = timedelta(0)
    test_time = timedelta(0)

    # Run the full learning rate schedule.
    for nepochs, learning_rate in zip(epochs, learning_rates):

        # Set end epoch for the current learning rate segment.
        end_epoch = start_epoch + nepochs

        # Set learning rate for the current segmenet.
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        _lr_optimizer = utils.get_learning_rate(optimizer)
        if _lr_optimizer is not None:
            print(f'Learning rate set to {_lr_optimizer}')
            assert _lr_optimizer == learning_rate

        # Run the current learning rate segment.
        for epoch in range(start_epoch, end_epoch):
            # Run an epoch of training.
            global_step, train_accs, epoch_train_time = run_epoch(
                epoch, global_step, model, train_loader, device,
                criterion=criterion, optimizer=optimizer,
                output_file=train_results_file, train=True,
                batch_callback=batch_callback, fp16=fp16)
            if results_file is not None:
                # Save high-level summary on training for the epoch.
                save_summary(epoch, global_step, train_accs, epoch_train_time,
                             results_file, 'train')
            train_time += epoch_train_time
            if train_accs[0].avg > best_train_accuracy:
                best_train_accuracy = train_accs[0].avg

            if dev_loader is not None:
                # Run validation
                _, dev_accs, epoch_dev_time = run_epoch(
                    epoch, global_step, model, dev_loader, device,
                    output_file=dev_results_file, train=False, fp16=fp16)
                if results_file is not None:
                    # Save high-level summary on validation for the epoch.
                    save_summary(epoch, global_step, dev_accs, epoch_dev_time,
                                 results_file, 'dev')
                dev_time += epoch_dev_time
                if dev_accs[0].avg > best_dev_accuracy:
                    best_dev_accuracy = dev_accs[0].avg

            if test_loader is not None:
                # Run testing
                _, test_accs, epoch_test_time = run_epoch(
                    epoch, global_step, model, test_loader, device,
                    output_file=test_results_file, train=False,
                    label='(Test): ', fp16=fp16)
                if results_file is not None:
                    # Save high-level summary on testing for the epoch.
                    save_summary(epoch, global_step, test_accs,
                                 epoch_test_time, results_file, 'test')
                test_time += epoch_test_time
                if test_accs[0].avg > best_test_accuracy:
                    best_test_accuracy = test_accs[0].avg

            # Check if performance has improved for early stopping.
            current_accuracy = train_accs[0].avg
            accuracy_mode = 'train'
            if dev_loader is not None:
                current_accuracy = dev_accs[0].avg
                accuracy_mode = 'dev'
            is_best = current_accuracy > best_accuracy
            if is_best:
                print(f'New best model! ({current_accuracy:0.2f})')
                best_accuracy = current_accuracy

            if run_dir is not None:
                # Check to see if a checkpoint should be created.
                last_epoch = epoch == (end_epoch - 1)
                is_last_checkpoint = (checkpoint == 'last' and last_epoch)
                should_checkpoint = is_best or is_last_checkpoint
                if not should_checkpoint:
                    should_checkpoint |= checkpoint == 'all'

                if should_checkpoint:
                    # Capture metadata and state of optimizer.
                    state = {
                        'epoch': epoch,
                        'accuracy': current_accuracy,
                        'mode': accuracy_mode,
                        'optimizer': optimizer.state_dict()
                    }

                    # Capture state of model.
                    if isinstance(model, nn.DataParallel):
                        # Get the state of the module so it can be
                        #   loaded on a CPU-only machine
                        state['model'] = model.module.state_dict()
                    else:
                        state['model'] = model.state_dict()

                    if is_last_checkpoint or checkpoint == 'all':
                        # Save metadata and state to disk.
                        checkpoint_path = os.path.join(
                            run_dir, f'checkpoint_{epoch}.t7')
                        print(f'Saving checkpoint to {checkpoint_path}')
                        torch.save(state, checkpoint_path)

                    if is_best:
                        # Save metadata and state to disk.
                        best_path = os.path.join(
                            run_dir, 'checkpoint_best_model.t7')
                        torch.save(state, best_path)

        # Update state epoch for next learning rate segment.
        start_epoch = end_epoch
    accuracy_splits = AccuracySplits(best_train_accuracy,
                                     best_dev_accuracy,
                                     best_test_accuracy)
    time_splits = TimeSplits(train_time, dev_time, test_time)
    return model, accuracy_splits, time_splits


def save_summary(epoch: int, global_step: int,
                 accuracies: List[utils.AverageMeter], duration: timedelta,
                 tracking_file: str, mode: str,
                 top=(1,)):
    result: Dict[str, Any] = OrderedDict()
    result['timestamp'] = datetime.now()
    result['mode'] = mode
    result['epoch'] = epoch
    result['global_step'] = global_step
    result['duration'] = duration

    for k, acc in zip(top, accuracies):
        result[f'top{k}_accuracy'] = acc.avg
    utils.save_result(result, tracking_file)


def run_epoch(epoch: int, global_step: int, model: nn.Module,
              loader: DataLoader, device: torch.device,
              criterion: Optional[nn.Module] = None,
              optimizer: Optional[Optimizer] = None,
              top: Tuple[int, ...] = (1,),
              output_file: Optional[str] = None,
              train: bool = True, label: Optional[str] = None,
              batch_callback: Optional[Callable] = None,
              fp16: bool = False
              ) -> Tuple[int, List[utils.AverageMeter], timedelta]:
    """
    Run a single epoch of train or validation

    Parameters
    ----------
    epoch : int
        Current epoch of training
    global_step : int
        Current step in training (i.e., `epoch * (len(train_loader))`)
    model : nn.Module
        Pytorch model to train (must support residual outputs)
    loader : DataLoader
        Training or validation data
    device : torch.device
        Device to load inputs and targets
    criterion : nn.Module or None, default None
        Loss function to optimize for (training only)
    optimizer : Optimizer or None, default None
        Optimizer to optimize model (training only)
    top : Tuple[int]
        Specify points to calculate accuracies (e.g., top-1 & top-5 -> (1, 5))
    output_file : str
        File path to log results
    train : bool
        Indicate whether to train the model and update weights
    label : str or None, default None
        Label for tqdm output
    batch_callback : Callable or None, default None
        Optional function to calculate stats on each batch.
    fp16 : bool, default False.
        Use mixed precision training.

    Returns
    -------
    global_step : int
    accuracy : float
        Top-{top[0]} accuracy from the combined network
    """
    if label is None:
        label = '(Train):' if train else '(Dev):'
    if train:
        assert criterion is not None, 'Need criterion to train model'
        _criterion = criterion
        assert optimizer is not None, 'Need optimizer to train model'
        _optimizer = optimizer
        losses = utils.AverageMeter()
        if fp16:
            from apex import amp  # avoid dependency unless necessary.

    accuracies = [utils.AverageMeter() for _ in top]
    wrapped_loader = tqdm(loader)
    model.train(train)

    if output_file is not None:
        write_heading = not os.path.exists(output_file)
    with maybe_open(output_file) as out_file:
        with torch.set_grad_enabled(train):
            start = datetime.now()
            total_time = timedelta(0)
            for batch_index, (indices, inputs, targets) in enumerate(wrapped_loader):  # noqa: E501
                batch_size = targets.size(0)
                assert batch_size < 2**32, 'Size is too large! correct will overflow'  # noqa: E501

                targets = targets.to(device)
                outputs = model(inputs)

                if batch_callback is not None:
                    # Allow selection metrics like forgetting events
                    batch_callback(indices, inputs, targets, outputs)

                if train:
                    global_step += 1
                    loss = _criterion(outputs, targets)
                    _optimizer.zero_grad()
                    if fp16:
                        with amp.scale_loss(loss, _optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    _optimizer.step()
                    losses.update(loss.item(), batch_size)
                top_correct = utils.correct(outputs, targets, top=top)
                for i, count in enumerate(top_correct):
                    accuracies[i].update(count.item() * (100. / batch_size), batch_size)  # noqa: E501
                end = datetime.now()  # Don't count logging overhead
                duration = end - start
                total_time += duration

                if output_file is not None:
                    result: Dict[str, Any] = OrderedDict()
                    result['timestamp'] = datetime.now()
                    result['batch_duration'] = duration
                    result['global_step'] = global_step
                    result['epoch'] = epoch
                    result['batch'] = batch_index
                    result['batch_size'] = batch_size
                    for i, k in enumerate(top):
                        result[f'top{k}_correct'] = top_correct[i].item()
                        result[f'top{k}_accuracy'] = accuracies[i].val
                    if train:
                        result['loss'] = loss.item()
                    utils.save_result(result, out_file,
                                      write_heading=write_heading)
                    write_heading = False

                desc = 'Epoch {} {}'.format(epoch, label)
                if train:
                    desc += ' Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)  # noqa: E501
                for k, acc in zip(top, accuracies):
                    desc += ' Top-{} {acc.val:.3f} ({acc.avg:.3f})'.format(k, acc=acc)  # noqa: E501
                wrapped_loader.set_description(desc, refresh=False)
                start = datetime.now()

    return global_step, accuracies, total_time


def maybe_open(path: Optional[str]):
    """
    Open file if the str is passed.

    Parameters
    ----------
    path : str or None
        File to open.

    Returns
    -------
    context : context manager
    """
    if path is not None:
        return open(path, mode='a')
    return ExitStack()
