import os
from collections import OrderedDict
from typing import Tuple, Optional, Callable, Dict, Any

import numpy as np
from torch import nn, cuda

from svp.common import utils
from svp.amazon.datasets import create_dataset
from svp.amazon.train import create_model_and_optimizer
from svp.common.train import run_training, create_loaders
from svp.common.selection import select
from svp.common.coreset import validate_splits, ForgettingEventsMeter


def coreset(run_dir: str = './run',

            datasets_dir: str = './data',
            dataset: str = 'amazon_review_polarity',
            validation: int = 0, shuffle: bool = True,

            arch: str = 'vdcnn9-maxpool', optimizer: str = 'sgd',
            epochs: Tuple[int, ...] = (3, 3, 3, 3, 3),
            learning_rates: Tuple[float, ...] = (
                0.01, 0.005, 0.0025, 0.00125, 0.000625
            ),
            momentum: float = 0.9, weight_decay: float = 1e-4,
            batch_size: int = 128, eval_batch_size: int = 128,

            proxy_arch: str = 'preact20', proxy_optimizer: str = 'sgd',
            proxy_epochs: Tuple[int, ...] = (3, 3, 3, 3, 3),
            proxy_learning_rates: Tuple[float, ...] = (
                0.01, 0.005, 0.0025, 0.00125, 0.000625
            ),
            proxy_momentum: float = 0.9, proxy_weight_decay: float = 1e-4,
            proxy_batch_size: int = 128, proxy_eval_batch_size: int = 128,

            subset: int = 1_800_000,
            selection_method: str = 'least_confidence',
            precomputed_selection: Optional[str] = None,
            train_target: bool = True,

            cuda: bool = True,
            device_ids: Tuple[int, ...] = tuple(range(cuda.device_count())),
            num_workers: int = 0, eval_num_workers: int = 0,

            seed: Optional[int] = None, checkpoint: str = 'best',
            track_test_acc: bool = True):
    """
    Perform core-set selection on Amazon Review Polarity and Full.

    Parameters
    ----------
    run_dir : str, default './run'
        Path to log results and other artifacts.
    datasets_dir : str, default './data'
        Path to datasets.
    dataset : str, default 'cifar10'
        Dataset to use in experiment (i.e., CIFAR10 or CIFAR100)
    validation : int, default 0
        Number of examples from training set to use for valdiation.
    shuffle : bool, default True
        Shuffle training data before splitting into training and validation.
    arch : str, default 'vdcnn9-maxpool'
        Model architecture for the target model. `vdcnn9-maxpool` is short for
        VDCNN9 with max pooling (see https://arxiv.org/abs/1606.01781).
    optimizer : str, default = 'sgd'
        Optimizer for training the target model.
    epochs : Tuple[int, ...], default (3, 3, 3, 3, 3)
        Epochs for training the target model. Each number corresponds to a
        learning rate below.
    learning_rates : Tuple[float, ...], default (
            0.01, 0.005, 0.0025, 0.00125, 0.000625)
        Learning rates for training the target model. Each learning rate is
        used for the corresponding number of epochs above.
    momentum : float, default 0.9
        Momentum for SGD with the target model.
    weight_decay : float, default 1e-4
        Weight decay for SGD with the target model.
    batch_size : int, default 128
        Minibatch size for training the target model.
    eval_batch_size : int, default 128
        Minibatch size for evaluation (validation and testing) of the target
        model.
    proxy_arch : str, default 'vdcnn9-maxpool'
        Model architecture for the proxy model. `vdcnn9-maxpool` is short for
        VDCNN9 with max pooling (see https://arxiv.org/abs/1606.01781).
    proxy_optimizer : str, default = 'sgd'
        Optimizer for training the proxy model.
    proxy_epochs : Tuple[int, ...], default (3, 3, 3, 3, 3)
        Epochs for training the proxy model. Each number corresponds to a
        learning rate below.
    proxy_learning_rates : Tuple[float, ...], default (
            0.01, 0.005, 0.0025, 0.00125, 0.000625)
        Learning rates for training the proxy model. Each learning rate is
        used for the corresponding number of epochs above.
    proxy_momentum : float, default 0.9
        Momentum for SGD with the proxy model.
    proxy_weight_decay : float, default 1e-4
        Weight decay for SGD with the proxy model.
    proxy_batch_size : int, default 128
        Minibatch size for training the proxy model.
    proxy_eval_batch_size : int, default 128
        Minibatch size for evaluation (validation and testing) of the proxy
        model.
    subset : int, default 1,800,000
        Number of examples to keep in the selected subset.
    selection_method : str, default least_confidence
        Criteria for selecting examples.
    precomputed_selection : str or None, default None
        Path to timestamped run_dir of precomputed indices.
    train_target : bool, default True
        If proxy and target are different, train the target after selection.
    cuda : bool, default True
        Enable or disable use of available GPUs
    device_ids : Tuple[int, ...], default True
        GPU device ids to use.
    num_workers : int, default 0
        Number of data loading workers for training.
    eval_num_workers : int, default 0
        Number of data loading workers for evaluation.
    seed : Optional[int], default None
        Random seed for numpy, torch, and others. If None, a random int is
        chosen and logged in the experiments config file.
    checkpoint : str, default 'best'
        Specify when to create a checkpoint for the model: only checkpoint the
        best performing model on the validation data or the training data if
        `validation == 0` ("best"), after every epoch ("all"), or only the last
        epoch of each segment of the learning rate schedule ("last").
    track_test_acc : bool, default True
        Calculate performance of the models on the test data in addition or
        instead of the validation dataset.'
    """
    # Set seeds for reproducibility.
    seed = utils.set_random_seed(seed)
    # Capture all of the arguments to save alongside the results.
    config = utils.capture_config(**locals())
    # Create a unique timestamped directory for this experiment.
    run_dir = utils.create_run_dir(run_dir, timestamp=config['timestamp'])
    utils.save_config(config, run_dir)
    # Update the computing arguments based on the runtime system.
    use_cuda, device, device_ids, num_workers = utils.config_run_env(
            cuda=cuda, device_ids=device_ids, num_workers=num_workers)

    # Create the training dataset.
    train_dataset = create_dataset(dataset, datasets_dir, train=True)
    # Verify there is enough training data for validation and
    #   the final selected subset.
    validate_splits(train_dataset, validation, subset)

    # Create the test dataset.
    test_dataset = None
    if track_test_acc:
        test_dataset = create_dataset(dataset, datasets_dir, train=False)

    # Calculate the number of classes (e.g., 10 or 100) so the model has
    #   the right dimension for its output.
    num_classes = train_dataset.classes  # type: ignore

    # Create the proxy and use it to select which data points should be
    #   used to train the final target model. If the selections were
    #   precomputed in another run or elsewhere, we can ignore this
    #   step.
    if precomputed_selection is None:
        # Create a directory for the proxy results to avoid confusion.
        proxy_run_dir = os.path.join(run_dir, 'proxy')
        os.makedirs(proxy_run_dir, exist_ok=True)

        # Split the training dataset between training and validation.
        train_indices, dev_indices = utils.split_indices(
            train_dataset, validation, proxy_run_dir, shuffle=shuffle)
        # Create data loaders for training, validation, and testing.
        train_loader, dev_loader, test_loader = create_loaders(
            train_dataset,
            batch_size=proxy_batch_size,
            eval_batch_size=proxy_eval_batch_size,
            test_dataset=test_dataset,
            use_cuda=use_cuda,
            num_workers=num_workers,
            eval_num_workers=eval_num_workers,
            indices=(train_indices, dev_indices))

        # Create the model and optimizer for training.
        model, _proxy_optimizer = create_model_and_optimizer(
            arch=proxy_arch,
            num_classes=num_classes,
            optimizer=proxy_optimizer,
            learning_rate=proxy_learning_rates[0],
            momentum=proxy_momentum,
            weight_decay=proxy_weight_decay,
            run_dir=proxy_run_dir)

        # Create the loss criterion.
        criterion = nn.CrossEntropyLoss()

        # Move the model and loss to the appropriate devices.
        model = model.to(device)
        if use_cuda:
            model = nn.DataParallel(model, device_ids=device_ids)
        criterion = criterion.to(device)

        # Potentially, create a callback to calculate selection stats
        #   during traing (e.g., forgetting events).
        batch_callback: Optional[Callable] = None
        # Only forgetting events is supported, but other things are
        #   fairly straightforward to add.
        if selection_method == 'forgetting_events':
            forgetting_meter = ForgettingEventsMeter(train_dataset)
            batch_callback = forgetting_meter.callback

        # Train the proxy model.
        model, proxy_accuracies, proxy_times = run_training(
            model=model,
            optimizer=_proxy_optimizer,
            criterion=criterion,
            device=device,
            train_loader=train_loader,
            epochs=proxy_epochs,
            learning_rates=proxy_learning_rates,
            dev_loader=dev_loader,
            test_loader=test_loader,
            run_dir=proxy_run_dir,
            checkpoint=checkpoint,
            batch_callback=batch_callback)
        # For analysis, record details about training the proxy.
        proxy_stats: Dict[str, Any] = OrderedDict()
        proxy_stats['nexamples'] = len(train_indices)
        proxy_stats['train_accuracy'] = proxy_accuracies.train
        proxy_stats['dev_accuracy'] = proxy_accuracies.dev
        proxy_stats['test_accuracy'] = proxy_accuracies.test

        proxy_stats['train_time'] = proxy_times.train
        proxy_stats['dev_time'] = proxy_times.dev
        proxy_stats['test_time'] = proxy_times.test
        utils.save_result(proxy_stats, os.path.join(run_dir, "proxy.csv"))

        current = np.array([], dtype=np.int64)
        # Create initial random subset for greedy k-center method.
        #   Everything else can start with an empty selected subset.
        if selection_method == 'kcenters':
            assert subset > 1_000
            # TODO: Maybe this shouldn't be hardcoded
            current = np.random.permutation(train_indices)[:1_000]

        nevents = None
        if selection_method == 'forgetting_events':
            nevents = forgetting_meter.nevents
            # Set the number of forgetting events for examples that the
            #   model never got correct to infinity as in the original
            #   paper.
            nevents[~forgetting_meter.was_correct] = np.inf
        # Select which data points should be used to train the final
        #   target model.
        target_train_indices, stats = select(model, train_dataset,
                                             current=current,
                                             pool=train_indices,
                                             budget=subset,
                                             method=selection_method,
                                             batch_size=proxy_eval_batch_size,
                                             device=device,
                                             device_ids=device_ids,
                                             num_workers=num_workers,
                                             use_cuda=use_cuda,
                                             nevents=nevents)
        # Save details for future runs and analysis.
        utils.save_index(target_train_indices, run_dir, 'selected.index')
        utils.save_index(dev_indices, run_dir, 'dev.index')
        utils.save_result(stats, os.path.join(run_dir, 'selection.csv'))

    else:
        assert train_target, "Must train target if selection is precomuted"
        # Read selected subset from previous run or elsewhere.
        target_train_indices = np.loadtxt(
            os.path.join(precomputed_selection, 'selected.index'),
            dtype=np.int64)
        dev_indices = np.loadtxt(
            os.path.join(precomputed_selection, 'dev.index'),
            dtype=np.int64)

    if train_target:
        # Create data loaders for training, validation, and testing.
        loaders = create_loaders(
            train_dataset,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            test_dataset=test_dataset,
            use_cuda=use_cuda,
            num_workers=num_workers,
            eval_num_workers=eval_num_workers,
            indices=(target_train_indices, dev_indices))
        target_train_loader, target_dev_loader, target_test_loader = loaders

        # Create a directory for the target results to avoid confusion.
        target_run_dir = os.path.join(run_dir, 'target')
        os.makedirs(target_run_dir, exist_ok=True)
        # Save details for future runs and analysis.
        utils.save_index(target_train_indices, target_run_dir, 'train.index')
        utils.save_index(dev_indices, target_run_dir, 'dev.index')

        # Create the model and optimizer for training.
        model, _target_optimizer = create_model_and_optimizer(
            arch=arch,
            num_classes=num_classes,
            optimizer=optimizer,
            learning_rate=learning_rates[0],
            momentum=momentum,
            weight_decay=weight_decay,
            run_dir=target_run_dir)

        # Create the loss criterion.
        criterion = nn.CrossEntropyLoss()

        # Move the model and loss to the appropriate devices.
        model = model.to(device)
        if use_cuda:
            model = nn.DataParallel(model, device_ids=device_ids)
        criterion = criterion.to(device)

        # Train the target model on the selected subset.
        model, target_accuracies, target_times = run_training(
            model=model,
            optimizer=_target_optimizer,
            criterion=criterion,
            device=device,
            train_loader=target_train_loader,
            epochs=epochs,
            learning_rates=learning_rates,
            dev_loader=target_dev_loader,
            test_loader=target_test_loader,
            run_dir=target_run_dir,
            checkpoint=checkpoint)
        # Save details for easy access and analysis.
        target_stats: Dict[str, Any] = OrderedDict()
        target_stats['nexamples'] = len(target_train_indices)
        target_stats['train_accuracy'] = target_accuracies.train
        target_stats['dev_accuracy'] = target_accuracies.dev
        target_stats['test_accuracy'] = target_accuracies.test

        target_stats['train_time'] = target_times.train
        target_stats['dev_time'] = target_times.dev
        target_stats['test_time'] = target_times.test
        utils.save_result(target_stats, os.path.join(run_dir, "target.csv"))
