import os
from typing import Tuple, Optional

import numpy as np
from torch import cuda
from torch import nn, optim
from torch.optim import Optimizer  # type: ignore

from svp.common import utils
from svp.imagenet.models import MODELS
from svp.imagenet.datasets import create_dataset
from svp.common.train import run_training, create_loaders


def train(run_dir: str = './run',

          datasets_dir: str = './data', dataset: str = 'imagenet',
          augmentation: bool = True,
          validation: int = 0, shuffle: bool = True,

          arch: str = 'resnet18', optimizer: str = 'sgd',
          epochs: Tuple[int, ...] = (1, 1, 1, 1, 1, 25, 30, 20, 20),
          learning_rates: Tuple[float, ...] = (
              0.0167, 0.0333, 0.05, 0.0667, 0.0833,  0.1, 0.01, 0.001, 0.0001),
          scale_learning_rates: bool = True,
          momentum: float = 0.9, weight_decay: float = 1e-4,
          batch_size: int = 256, eval_batch_size: int = 256,
          fp16: bool = False, label_smoothing: float = 0.1,
          loss_scale: float = 256.0,

          cuda: bool = True,
          device_ids: Tuple[int, ...] = tuple(range(cuda.device_count())),
          num_workers: int = 0, eval_num_workers: int = 0,

          seed: Optional[int] = None, checkpoint: str = 'best',
          track_test_acc: bool = True):
    """
    Train deep learning models (e.g., ResNet) on ImageNet.

    Parameters
    ----------
    run_dir : str, default './run'
        Path to log results and other artifacts.
    datasets_dir : str, default './data'
        Path to datasets.
    dataset : str, default 'imagenet'
        Dataset to use in experiment (unnecessary but kept for consistency)
    augmentation : bool, default True
        Add data augmentation (i.e., random crop and horizontal flip).
    validation : int, default 0
        Number of examples from training set to use for valdiation.
    shuffle : bool, default True
        Shuffle training data before splitting into training and validation.
    arch : str, default 'resnet18'
        Model architecture. `resnet18` is short for ResNet18. Other models are
        pulled from `torchvision.models`.
    optimizer : str, default = 'sgd'
        Optimizer for training.
    epochs : Tuple[int, ...], default (1, 1, 1, 1, 1, 25, 30, 20, 20)
        Epochs for training. Each number corresponds to a learning rate below.
    learning_rates : Tuple[float, ...], default (
            0.0167, 0.0333, 0.05, 0.0667, 0.0833,  0.1, 0.01, 0.001, 0.0001)
        Learning rates for training. Each learning rate is used for the
        corresponding number of epochs above.
    scale_learning_rates : bool, default True
        Scale learning rates above based on (`batch_size / 256`). Mainly for
        convenience with large minibatch training.
    momentum : float, default 0.9
        Momentum for SGD.
    weight_decay : float, default 1e-4
        Weight decay for SGD.
    batch_size : int, default 256
        Minibatch size for training.
    eval_batch_size : int, default 256
        Minibatch size for evaluation (validation and testing).
    fp16 : bool, default False
        Use mixed precision training.
    label_smoothing : float, default 0.1
        Amount to smooth labels for loss.
    loss_scale : float, default 256
        Amount to scale loss for mixed precision training.
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
    if scale_learning_rates:
        # For convenience, scale the learning rate for large-batch SGD
        learning_rates = tuple(np.array(learning_rates) * (batch_size / 256))
        config['scaled_learning_rates'] = learning_rates
    # Create a unique timestamped directory for this experiment.
    run_dir = utils.create_run_dir(run_dir, timestamp=config['timestamp'])
    utils.save_config(config, run_dir)
    # Update the computing arguments based on the runtime system.
    use_cuda, device, device_ids, num_workers = utils.config_run_env(
            cuda=cuda, device_ids=device_ids, num_workers=num_workers)

    # Create the training dataset.
    train_dataset = create_dataset(dataset, datasets_dir, train=True,
                                   augmentation=augmentation)
    # Create the test dataset.
    test_dataset = None
    if track_test_acc:
        test_dataset = create_dataset(dataset, datasets_dir, train=False,
                                      augmentation=False)

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

    # Calculate the number of classes (e.g., 1000) so the model has
    #   the right dimension for its output.
    num_classes = 1_000  # type: ignore

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
    criterion = _LabelSmoothing(label_smoothing)

    # Move the model and loss to the appropriate devices.
    model = model.to(device)
    criterion = criterion.to(device)

    if fp16:
        from apex import amp  # avoid dependency unless necessary.
        model, _optimizer = amp.initialize(model, _optimizer,
                                           loss_scale=loss_scale)
    if use_cuda:
        model = nn.DataParallel(model, device_ids=device_ids)

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
        fp16=fp16,
        run_dir=run_dir,
        checkpoint=checkpoint)


def create_model_and_optimizer(arch: str, num_classes: int, optimizer: str,
                               learning_rate: float,
                               momentum: Optional[float] = None,
                               weight_decay: Optional[float] = None,
                               run_dir: Optional[str] = None
                               ) -> Tuple[nn.Module, Optimizer]:
    '''
    Create the model and optimizer for ImageNet.

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


class _LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(_LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
