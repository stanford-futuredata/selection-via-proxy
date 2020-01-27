import os
from typing import Tuple, Optional

import torch
import click
from torch import nn, optim
from torch.optim import Optimizer  # type: ignore
import torch.backends.cudnn as cudnn

from svp.common import utils
from svp.cifar.models import MODELS
from svp.cifar.datasets import DATASETS, create_dataset
from svp.common.train import run_training, load_and_split


@click.command()
@click.option('--run-dir', default='./run', show_default=True,
              help='Path to log results and other artifacts.')
# Dataset options
@click.option('--datasets-dir', default='./data', show_default=True,
              help='Path to datasets.')
@click.option('--dataset', '-d', type=click.Choice(DATASETS),
              default='cifar10', show_default=True,
              help='Specify dataset to use in experiment.')
@click.option('--augmentation/--no-augmentation',
              default=True, show_default=True,
              help='Add data augmentation.')
@click.option('--validation', '-v', default=0, show_default=True,
              help='Number of examples to use for valdiation')
@click.option('--shuffle/--no-shuffle', default=True, show_default=True,
              help='Shuffle train and validation data before splitting.')
# Training options
@click.option('--arch', '-a', type=click.Choice(MODELS.keys()),
              default='resnet20', show_default=True,
              help='Specify model archtiecture.')
@click.option('--optimizer', '-o', type=click.Choice(['sgd', 'adam']),
              default='sgd', show_default=True,
              help='Specify optimizer for training.')
@click.option('--epochs', '-e', multiple=True, type=int,
              default=(1, 90, 45, 45), show_default=True,
              help='Specify epochs for training.')
@click.option('learning_rates', '--learning-rate', '-l', multiple=True,
              type=float, default=(0.01, 0.1, 0.01, 0.001), show_default=True,
              help='Specify learning rate for training.')
@click.option('--momentum', type=float, default=0.9, show_default=True,
              help='Specify proxy momentum.')
@click.option('--weight-decay', type=float, default=1e-4, show_default=True,
              help='Specify weight decay.')
@click.option('--batch-size', '-b', default=128, show_default=True,
              help='Specify minibatch size for training.')
@click.option('--eval-batch-size', type=int, callback=utils.override_option,
              help='Override minibatch size for evaluation')
# Computing options
@click.option('--cuda/--no-cuda', default=True, show_default=True,
              help="Enable or disable available GPUs")
@click.option('device_ids', '--device', '-d', multiple=True, type=int,
              default=tuple(range(torch.cuda.device_count())),
              show_default=True,
              help="Specify device ids for GPUs to use.")
@click.option('--num-workers', type=int, default=0, show_default=True,
              help="Number of workers to use for data loading for training")
@click.option('--eval-num-workers', type=int, callback=utils.override_option,
              help="Number of workers to use for data loading for evaluation")
# MISC
@click.option('--seed', '-s', type=int,
              help='Random seed')
@click.option('--checkpoint', '-c', type=click.Choice(['best', 'all', 'last']),
              default='best', show_default=True,
              help='Specify when to checkpoint model: only the best '
                   ' performing model on the validation data ("best"),'
                   ' after every epoch ("all"), or'
                   ' only the last epoch of each segment of the learning'
                   ' rate schedule ("last").')
@click.option('--track-test-acc/--no-track-test-acc',
              default=True, show_default=True,
              help='Calculate performance of the models on the test in'
                   ' addition or instead of the validation dataset.')
def train(run_dir: str,

          datasets_dir: str, dataset: str, augmentation: bool,
          validation: int, shuffle: bool,

          arch: str, optimizer: str, epochs: Tuple[int, ...],
          learning_rates: Tuple[float, ...],
          momentum: float, weight_decay: float,
          batch_size: int, eval_batch_size: int,

          cuda: bool, device_ids: Tuple[int, ...],
          num_workers: int, eval_num_workers: int,

          seed: int, checkpoint: str, track_test_acc: bool):
    # Configure run's workspace
    seed = utils.set_random_seed(seed)  # set seeds for reproducibility
    config = utils.capture_config(**locals())
    run_dir = utils.create_run_dir(run_dir, timestamp=config['timestamp'])
    utils.save_config(config, run_dir)
    use_cuda, device, device_ids, num_workers = utils.config_run_env(
            cuda=cuda, device_ids=device_ids, num_workers=num_workers)

    # Create datasets
    train_dataset = create_dataset(dataset, datasets_dir, train=True,
                                   augmentation=augmentation)
    if augmentation:
        valid_dataset = create_dataset(dataset, datasets_dir, train=True,
                                       augmentation=False)
    else:
        valid_dataset = train_dataset
    test_dataset = create_dataset(dataset, datasets_dir, train=False,
                                  augmentation=False)
    assert len(train_dataset) == len(valid_dataset)

    # Create data loaders
    train_loader, valid_loader, test_loader = load_and_split(
        train_dataset, valid_dataset, test_dataset, validation, run_dir,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        use_cuda=use_cuda,
        shuffle=shuffle,
        num_workers=num_workers)

    num_classes = len(set(test_dataset.targets))  # type: ignore
    model, optimizer = create_graph(
        run_dir=run_dir,
        arch=arch,
        num_classes=num_classes,
        optimizer=optimizer,
        learning_rate=learning_rates[0],
        momentum=momentum,
        weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    if use_cuda:
        model = nn.DataParallel(model, device_ids=device_ids)
    criterion = criterion.to(device)

    run_training(
        run_dir=run_dir,
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


def create_graph(run_dir: str, arch: str, num_classes: int, optimizer: str,
                 learning_rate: float, momentum: Optional[float] = None,
                 weight_decay: Optional[float] = None
                 ) -> Tuple[nn.Module, Optimizer]:
    '''
    Args
    - run_dir: str, path to directory
    - arch: str, key in MODELS
    - num_classes: int
    - optimizer: str, one of ['adam', 'sgd']
    - learning_rate: float
    - momemtum: float, only used if optimizer='sgd'
    - weight_decay: float, only used if optimizer='sgd'
    Returns
    - model: torch.nn.Module
    - optimizer: torch.optim.Optimizer
    '''
    # create model
    model = MODELS[arch](num_classes=num_classes)

    # create optimizer
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

    # Save model text description
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))
    return model, _optimizer


if __name__ == '__main__':
    if torch.cuda.is_available():
        cudnn.benchmark = True  # type: ignore
    train()
