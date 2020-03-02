import os
from typing import Tuple, Optional

from torch import cuda
from torch import nn, optim
from torch.optim import Optimizer  # type: ignore

from svp.common import utils
from svp.common.train import run_training, create_loaders
from svp.amazon.models import MODELS
from svp.amazon.datasets import create_dataset


def train(run_dir: str = './run',

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

          cuda: bool = True,
          device_ids: Tuple[int, ...] = tuple(range(cuda.device_count())),
          num_workers: int = 0, eval_num_workers: int = 0,

          seed: Optional[int] = None, checkpoint: str = 'best',
          track_test_acc: bool = True):
    """
    Train deep learning models (e.g., VDCNN) on Amaon Review Polarity and Full.

    Parameters
    ----------
    run_dir : str, default './run'
        Path to log results and other artifacts.
    datasets_dir : str, default './data'
        Path to datasets.
    dataset : str, default 'amazon_review_polarity'
        Dataset to use in experiment (e.g., amazon_review_full)
    validation : int, default 0
        Number of examples from training set to use for valdiation.
    shuffle : bool, default True
        Shuffle training data before splitting into training and validation.
    arch : str, default 'vdcnn9-maxpool'
        Model architecture. `vdcnn9-maxpool` is short for VDCNN9 with max
        pooling (see https://arxiv.org/abs/1606.01781).
    optimizer : str, default = 'sgd'
        Optimizer for training.
    epochs : Tuple[int, ...], default (3, 3, 3, 3, 3)
        Epochs for training. Each number corresponds to a learning rate below.
    learning_rates : Tuple[float, ...], default (
            0.01, 0.005, 0.0025, 0.00125, 0.000625)
        Learning rates for training. Each learning rate is used for the
        corresponding number of epochs above.
    momentum : float, default 0.9
        Momentum for SGD.
    weight_decay : float, default 1e-4
        Weight decay for SGD.
    batch_size : int, default 128
        Minibatch size for training.
    eval_batch_size : int, default 128
        Minibatch size for evaluation (validation and testing).
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

    Returns
    -------
    model : nn.Module
        Trained model.
    accuracies : Tuple[float, ...]
        The best accuracies from the model on the train, dev, and test splits.
    times : Tuple[timedelta, ...]
        Time spent training or evaluating on the train, dev, and test splits.
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

    # Create the test dataset.
    test_dataset = None
    if track_test_acc:
        test_dataset = create_dataset(dataset, datasets_dir, train=False)

    # Create data loaders
    train_loader, dev_loader, test_loader = create_loaders(
        train_dataset,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        validation=validation,
        run_dir=run_dir,
        test_dataset=test_dataset,
        use_cuda=use_cuda,
        shuffle=shuffle,
        num_workers=num_workers,
        eval_num_workers=eval_num_workers)

    # Calculate the number of classes (e.g., 2 or 5) so the model has
    #   the right dimension for its output.
    num_classes = train_dataset.classes

    # Create the model and optimizer for training.
    model, _optimizer = create_model_and_optimizer(
        run_dir=run_dir,
        arch=arch,
        num_classes=num_classes,
        optimizer=optimizer,
        learning_rate=learning_rates[0],
        momentum=momentum,
        weight_decay=weight_decay)

    # Create the loss criterion.
    criterion = nn.CrossEntropyLoss()

    # Move the model and loss to the appropriate devices.
    model = model.to(device)
    if use_cuda:
        model = nn.DataParallel(model, device_ids=device_ids)
    criterion = criterion.to(device)

    # Run training.
    return run_training(
        model=model,
        optimizer=_optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        epochs=epochs,
        learning_rates=learning_rates,
        dev_loader=dev_loader,
        test_loader=test_loader,
        run_dir=run_dir,
        checkpoint=checkpoint)


def create_model_and_optimizer(arch: str, num_classes: int, optimizer: str,
                               learning_rate: float,
                               momentum: Optional[float] = None,
                               weight_decay: Optional[float] = None,
                               run_dir: Optional[str] = None
                               ) -> Tuple[nn.Module, Optimizer]:
    '''
    Create the model and optimizer for Amazon datasets.

    Parameters
    ----------
    arch : str
        Name of model architecture (i.e., key in MODELS).
    num_classes : int
        Number of output classes.
    optimizer : str
        Name of optimizer (i.e., 'adam' or 'sgd').
    learning_rate : float
        Initial learning rate for training.
    momemtum : float or None, default None
        Amount of momentum during training.
        Only used if `optimizer='sgd'`.
    weight_decay : float or None, default None
        Amount of weight decay as regularization.
        Only used if `optimizer='sgd'`.
    run_dir : str or None, default None.
        Path to logging directory.

    Returns
    -------
    model : torch.nn.Module
    optimizer : torch.optim.Optimizer
    '''
    # Create model
    model = MODELS[arch](num_classes=num_classes)

    # Create optimizer
    if optimizer == 'adam':
        _optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        assert weight_decay is not None, "SGD needs weight decay"
        assert momentum is not None, "SGD needs momentum"
        _optimizer = optim.SGD(  # type: ignore
            model.parameters(), lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay)
    else:
        raise NotImplementedError(f'Unknown optimizer: {optimizer}')

    if run_dir is not None:
        # Save model text description
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
            file.write(str(model))
    return model, _optimizer
