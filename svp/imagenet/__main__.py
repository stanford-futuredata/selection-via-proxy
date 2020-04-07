from typing import Tuple, Optional

import click
import torch
import torch.backends.cudnn as cudnn

from svp.common import utils
from svp.common.cli import computing_options, miscellaneous_options
from svp.common.active import SELECTION_METHODS as active_learning_methods
from svp.common.coreset import SELECTION_METHODS as coreset_methods
from svp.imagenet.models import MODELS
from svp.imagenet.datasets import DATASETS
from svp.imagenet.train import train as train_function
from svp.imagenet.active import active as active_function
from svp.imagenet.coreset import coreset as coreset_function


@click.group()
def cli():
    pass


def dataset_options(func):
    # Dataset options
    decorators = [
        click.option('--datasets-dir', default='./data', show_default=True,
                     help='Path to datasets.'),
        click.option('--dataset', '-d', type=click.Choice(DATASETS),
                     default='imagenet', show_default=True,
                     help='Specify dataset to use in experiment.'),
        click.option('--augmentation/--no-augmentation',
                     default=True, show_default=True,
                     help='Add data augmentation.'),
        click.option('--validation', '-v', default=0, show_default=True,
                     help='Number of examples to use for valdiation'),
        click.option('--shuffle/--no-shuffle', default=True, show_default=True,
                     help=('Shuffle train and validation data before'
                           ' splitting.'))
    ]
    decorators.reverse()
    for decorator in decorators:
        func = decorator(func)
    return func


def training_options(func):
    decorators = [
        click.option('--arch', '-a', type=click.Choice(MODELS.keys()),
                     default='resnet18', show_default=True,
                     help='Specify model architecture.'),
        click.option('--optimizer', '-o', type=click.Choice(['sgd', 'adam']),
                     default='sgd', show_default=True,
                     help='Specify optimizer for training.'),
        click.option('--epochs', '-e', multiple=True, type=int,
                     default=(1, 1, 1, 1, 1, 25, 30, 20, 20),
                     show_default=True,
                     help='Specify epochs for training.'),
        click.option('learning_rates', '--learning-rate', '-l', multiple=True,
                     type=float, show_default=True,
                     default=(
                        0.0167, 0.0333, 0.05, 0.0667, 0.0833,
                        0.1, 0.01, 0.001, 0.0001),
                     help='Specify learning rate for training.'),
        click.option('--scale-learning-rates/--no-scale-learning-rates',
                     default=True, show_default=True,
                     help='Scale learning rates based on (batch size / 256)'),
        click.option('--momentum', type=float, default=0.9, show_default=True,
                     help='Specify proxy momentum.'),
        click.option('--weight-decay', type=float, default=1e-4,
                     show_default=True,
                     help='Specify weight decay.'),
        click.option('--batch-size', '-b', default=256, show_default=True,
                     help='Specify minibatch size for training.'),
        click.option('--eval-batch-size', type=int,
                     callback=utils.override_option,
                     help='Override minibatch size for evaluation'),
        click.option('--fp16/--no-fp16', default=False, show_default=True,
                     help='Use mixed precision training'),
        click.option('--label-smoothing', default=0.1, show_default=True,
                     help='Amount to smooth labels for loss'),
        click.option('--loss-scale', default=256.0, show_default=True,
                     help='Amount to scale loss for mixed precision training')
    ]
    decorators.reverse()
    for decorator in decorators:
        func = decorator(func)
    return func


@cli.command()
@click.option('--run-dir', default='./run', show_default=True,
              help='Path to log results and other artifacts.')
@dataset_options
@training_options
@computing_options
@miscellaneous_options
def train(run_dir: str,

          datasets_dir: str, dataset: str, augmentation: bool,
          validation: int, shuffle: bool,

          arch: str, optimizer: str, epochs: Tuple[int, ...],
          learning_rates: Tuple[float, ...],
          scale_learning_rates: bool,
          momentum: float, weight_decay: float,
          batch_size: int, eval_batch_size: int,
          fp16: bool, label_smoothing: float, loss_scale: float,

          cuda: bool, device_ids: Tuple[int, ...],
          num_workers: int, eval_num_workers: int,

          seed: int, checkpoint: str, track_test_acc: bool):
    train_function(**locals())


def proxy_training_overrides(func):
    decorators = [
        click.option('--proxy-arch', '-a', type=click.Choice(MODELS.keys()),
                     callback=utils.override_option,
                     help='Override proxy model architecture.'),
        click.option('--proxy-optimizer', '-o',
                     type=click.Choice(['sgd', 'adam']),
                     callback=utils.override_option,
                     help='Specify optimizer for training.'),
        click.option('--proxy-epochs', multiple=True, type=int,
                     callback=utils.override_option,
                     help='Override epochs for proxy training.'),
        click.option('proxy_learning_rates', '--proxy-learning-rate',
                     multiple=True, type=float,
                     callback=utils.override_option,
                     help='Override proxy learning rate for training.'),
        click.option('--proxy-scale-learning-rates/--no-proxy-scale-learning-rates',  # noqa: E501
                     default=True, show_default=True,
                     help='Override learning rate scaling'),
        click.option('--proxy-momentum', type=float,
                     callback=utils.override_option,
                     help='Override momentum.'),
        click.option('--proxy-weight-decay', type=float,
                     callback=utils.override_option,
                     help='Override weight decay.'),
        click.option('--proxy-batch-size', type=int,
                     callback=utils.override_option,
                     help='Override proxy minibatch size for training.'),
        click.option('--proxy-eval-batch-size', type=int,
                     callback=utils.override_option,
                     help='Override proxy minibatch size for evaluation'),
        click.option('--proxy-fp16/--no-proxy-fp16',
                     default=False, show_default=True,
                     help='Override mixed precision training'),
        click.option('--proxy-label-smoothing', type=float,
                     callback=utils.override_option,
                     help='Override label smoothing'),
        click.option('--proxy-loss-scale', type=float,
                     callback=utils.override_option,
                     help='Override loss scaling')
    ]
    decorators.reverse()
    for decorator in decorators:
        func = decorator(func)
    return func


@cli.command()
@click.option('--run-dir', default='./run', show_default=True,
              help='Path to log results and other artifacts.')
@dataset_options
@training_options
@proxy_training_overrides
# Active learning options
@click.option('--initial-subset', type=int, default=25_623, show_default=True,
              help='Number of training examples to use for initial'
                   ' labelled set.')
@click.option('rounds', '--round', '-r', multiple=True, type=int,
              default=(102_493, 128_117, 128_117, 128_117, 128_117),
              show_default=True,
              help='Number of unlabelled examples to select in a round of'
                   ' labeling.')
@click.option('--selection-method', type=click.Choice(active_learning_methods),
              default='least_confidence', show_default=True,
              help='Criteria for selecting examples')
@click.option('--precomputed-selection',
              help='Path to timestamp run_dir of precomputed indices')
@click.option('--train-target/--no-train-target',
              default=True, show_default=True,
              help=('If proxy and target are different, train the target'
                    ' after each round of selection'))
@click.option('--eval-target-at', multiple=True, type=int,
              help=('If proxy and target are different and --train-target,'
                    ' limit the evaluation of the target model to specific'
                    ' labelled subset sizes'))
@computing_options
@miscellaneous_options
def active(run_dir: str,

           datasets_dir: str, dataset: str, augmentation: bool,
           validation: int, shuffle: bool,

           arch: str, optimizer: str,
           epochs: Tuple[int, ...],
           learning_rates: Tuple[float, ...],
           scale_learning_rates: bool,
           momentum: float, weight_decay: float,
           batch_size: int, eval_batch_size: int,
           fp16: bool, label_smoothing: float, loss_scale: float,

           proxy_arch: str, proxy_optimizer: str,
           proxy_epochs: Tuple[int, ...],
           proxy_learning_rates: Tuple[float, ...],
           proxy_scale_learning_rates: bool,
           proxy_momentum: float, proxy_weight_decay: float,
           proxy_batch_size: int, proxy_eval_batch_size: int,
           proxy_fp16: bool, proxy_label_smoothing: float,
           proxy_loss_scale: float,

           initial_subset: int,  rounds: Tuple[int, ...],
           selection_method: str, precomputed_selection: Optional[str],
           train_target: bool, eval_target_at: Tuple[int, ...],

           cuda: bool, device_ids: Tuple[int, ...],
           num_workers: int, eval_num_workers: int,

           seed: int, checkpoint: str, track_test_acc: bool):
    active_function(**locals())


@cli.command()
@click.option('--run-dir', default='./run', show_default=True,
              help='Path to log results and other artifacts.')
@dataset_options
@training_options
@proxy_training_overrides
# Core-set selection options
@click.option('--subset', type=int, default=512_466, show_default=True,
              help='Number of examples to keep in the selected subset.')
@click.option('--selection-method', type=click.Choice(coreset_methods),
              default='least_confidence', show_default=True,
              help='Criteria for selecting unlabelled examples to label')
@click.option('--precomputed-selection',
              help='Path to timestamp run_dir of precomputed indices')
@click.option('--train-target/--no-train-target',
              default=True, show_default=True,
              help=('If proxy and target are different, train the target'
                    ' after selection'))
@computing_options
@miscellaneous_options
def coreset(run_dir: str,

            datasets_dir: str, dataset: str, augmentation: bool,
            validation: int, shuffle: bool,

            arch: str, optimizer: str,
            epochs: Tuple[int, ...],
            learning_rates: Tuple[float, ...],
            scale_learning_rates: bool,
            momentum: float, weight_decay: float,
            batch_size: int, eval_batch_size: int,
            fp16: bool, label_smoothing: float, loss_scale: float,

            proxy_arch: str, proxy_optimizer: str,
            proxy_epochs: Tuple[int, ...],
            proxy_learning_rates: Tuple[float, ...],
            proxy_scale_learning_rates: bool,
            proxy_momentum: float, proxy_weight_decay: float,
            proxy_batch_size: int, proxy_eval_batch_size: int,
            proxy_fp16: bool, proxy_label_smoothing: float,
            proxy_loss_scale: float,

            subset: int, selection_method: str,
            precomputed_selection: Optional[str], train_target: bool,

            cuda: bool, device_ids: Tuple[int, ...],
            num_workers: int, eval_num_workers: int,

            seed: int, checkpoint: str, track_test_acc: bool):
    coreset_function(**locals())


if __name__ == '__main__':
    if torch.cuda.is_available():
        cudnn.benchmark = True  # type: ignore
    cli()
