import os
from datetime import datetime, timedelta
from contextlib import ExitStack
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict, Any

import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from svp.common import utils


def load_and_split(train_dataset: Dataset, valid_dataset: Dataset,
                   test_dataset: Dataset,
                   validation: int, run_dir: str,
                   batch_size: int = 128, eval_batch_size: int = 128,
                   shuffle: bool = False, use_cuda: bool = False,
                   num_workers: int = 0
                   ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test splits from datasets

    Parameters
    ----------
    train_dataset : Dataset
    valid_dataset : Dataset
    test_dataset : Dataset
    validation : int
    run_dir : str
    batch_size : int
    eval_batch_size : int
    shuffle : bool, default False
    use_cuda : bool, default False
    num_workers : int, default 0

    Returns
    -------
    train_loader : DataLoader
    valid_loader : DataLoader
    test_loader : DataLoader
    """
    train_indices, valid_indices = utils.split_indices(
        train_dataset, validation, run_dir, shuffle=shuffle)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size,
        num_workers=num_workers, pin_memory=use_cuda)

    if len(valid_indices) > 0:
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, sampler=valid_sampler, batch_size=eval_batch_size,
            num_workers=num_workers, pin_memory=use_cuda)
    else:
        print('Using test dataset for validation')
        valid_loader = test_loader
    return train_loader, valid_loader, test_loader


def run_training(run_dir: str, model: nn.Module,
                 optimizer: Optimizer, criterion: nn.Module,
                 device: torch.device, train_loader: DataLoader,
                 valid_loader: DataLoader, test_loader: DataLoader,
                 epochs: Tuple[int, ...], learning_rates: Tuple[float, ...],
                 checkpoint: str, use_cuda: bool, track_test_acc: bool = True
                 ) -> Tuple[nn.Module, float, timedelta, timedelta]:
    """
    Run training with evaluation after every epoch

    Parameters
    ----------
    run_dir : str
        Directory to save results from training and evaluation
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
    valid_loader : DataLoader
        Validation data loader (also used for early stopping)
    test_loader : DataLoader
        Test data loader
    epochs : Tuple[int]
        Number of epochs for each learning rate
    learning_rates : Tuple[float]
        Learning rates for optimizer
    checkpoint : str
        Type of checkpointing to perform
    use_cuda : bool
        Indicate whether to use GPUs
    track_test_acc : bool, default True
        Indicate whether to track test accuracy
    """
    assert len(epochs) == len(learning_rates)
    start_epoch = 1
    global_step = 0
    best_accuracy = 0.0
    train_results_file = os.path.join(run_dir, 'train_results.csv')
    valid_results_file = os.path.join(run_dir, 'valid_results.csv')
    test_results_file = os.path.join(run_dir, 'test_results.csv')
    results_file = os.path.join(run_dir, 'results.csv')

    for nepochs, learning_rate in zip(epochs, learning_rates):
        end_epoch = start_epoch + nepochs
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        _lr_optimizer = utils.get_learning_rate(optimizer)
        if _lr_optimizer is not None:
            print(f'Learning rate set to {_lr_optimizer}')
            assert _lr_optimizer == learning_rate

        train_time = timedelta(0)
        eval_time = timedelta(0)
        for epoch in range(start_epoch, end_epoch):
            global_step, train_accs, epoch_train_time = run_epoch(
                epoch, global_step, model, train_loader, device,
                criterion=criterion, optimizer=optimizer,
                output_file=train_results_file, train=True)
            save_summary(epoch, global_step, train_accs, results_file, 'train')
            train_time += epoch_train_time

            _, valid_accs, epoch_eval_time = run_epoch(
                epoch, global_step, model, valid_loader, device,
                output_file=valid_results_file, train=False)
            save_summary(epoch, global_step, valid_accs, results_file, 'valid')
            eval_time += epoch_eval_time

            if valid_loader != test_loader and track_test_acc:
                _, test_accs, _ = run_epoch(
                    epoch, global_step, model, test_loader, device,
                    output_file=test_results_file, train=False,
                    label='(Test): ')
                save_summary(epoch, global_step, test_accs, results_file,
                             'test')

            valid_acc = valid_accs[0].avg
            is_best = valid_acc > best_accuracy
            last_epoch = epoch == (end_epoch - 1)
            if is_best or checkpoint == 'all' or (checkpoint == 'last' and last_epoch):  # noqa: E501
                state = {
                    'epoch': epoch,
                    # TODO: add extra parameters for model
                    'model': (model.module if use_cuda else model).state_dict(),  # noqa: E501
                    'accuracy': valid_acc,
                    'optimizer': optimizer.state_dict()
                }
            if is_best:
                print(f'New best model! ({valid_acc:0.2f})')
                filename = os.path.join(run_dir, 'checkpoint_best_model.t7')
                print(f'Saving checkpoint to {filename}')
                best_accuracy = valid_acc
                torch.save(state, filename)
            if checkpoint == 'all' or (checkpoint == 'last' and last_epoch):
                filename = os.path.join(run_dir, f'checkpoint_{epoch}.t7')
                print(f'Saving checkpoint to {filename}')
                torch.save(state, filename)

        start_epoch = end_epoch
    return model, best_accuracy, train_time, eval_time


def save_summary(epoch, global_step, accuracies, tracking_file, mode,
                 top=(1,)):
    result: Dict[str, Any] = OrderedDict()
    result['timestamp'] = datetime.now()
    result['mode'] = mode
    result['epoch'] = epoch
    result['global_step'] = global_step

    for k, acc in zip(top, accuracies):
        result[f'top{k}_accuracy'] = acc.avg
    utils.save_result(result, tracking_file)


def run_epoch(epoch: int, global_step: int, model: nn.Module,
              loader: DataLoader, device: torch.device,
              criterion: Optional[nn.Module] = None,
              optimizer: Optional[Optimizer] = None,
              top: Tuple[int, ...] = (1,),
              output_file: Optional[str] = None,
              train: bool = True, label: Optional[str] = None
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

    Returns
    -------
    global_step : int
    accuracy : float
        Top-{top[0]} accuracy from the combined network
    """
    if label is None:
        label = '(Train):' if train else '(Valid):'
    if train:
        assert criterion is not None, 'Need criterion to train model'
        _criterion = criterion
        assert optimizer is not None, 'Need optimizer to train model'
        _optimizer = optimizer
        losses = utils.AverageMeter()

    accuracies = [utils.AverageMeter() for _ in top]
    wrapped_loader = tqdm(loader)
    model.train(train)

    if output_file is not None:
        write_heading = not os.path.exists(output_file)
    with maybe_open(output_file) as out_file:
        with torch.set_grad_enabled(train):
            start = datetime.now()
            total_time = timedelta(0)
            for batch_index, (inputs, targets) in enumerate(wrapped_loader):
                batch_size = targets.size(0)
                assert batch_size < 2**32, 'Size is too large! correct will overflow'  # noqa: E501

                targets = targets.to(device)
                outputs = model(inputs)

                if train:
                    global_step += 1
                    loss = _criterion(outputs, targets)
                    _optimizer.zero_grad()
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


def maybe_open(output_file: Optional[str]):
    """
    Open file if the str is passed.
    """
    if output_file is not None:
        return open(output_file, mode='a')
    return ExitStack()
