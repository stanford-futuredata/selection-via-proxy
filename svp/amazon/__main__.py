from typing import Tuple, Optional

import torch
import click
import torch.backends.cudnn as cudnn

from svp.common import utils
from svp.common.cli import computing_options, miscellaneous_options
from svp.common.active import SELECTION_METHODS as active_learning_methods
from svp.common.coreset import SELECTION_METHODS as coreset_methods
from svp.amazon.models import MODELS
from svp.amazon.datasets import DATASETS
from svp.amazon.train import train as train_function
from svp.amazon.active import active as active_function
from svp.amazon.coreset import coreset as coreset_function
from svp.amazon.fasttext import fasttext as fasttext_function
from svp.amazon.fasttext import FASTTEXT_SELECTION_METHODS


@click.group()
def cli():
    pass


def dataset_options(func):
    # Dataset options
    decorators = [
        click.option('--datasets-dir', default='./data', show_default=True,
                     help='Path to datasets.'),
        click.option('--dataset', '-d', type=click.Choice(DATASETS),
                     default='amazon_review_polarity', show_default=True,
                     help='Specify dataset to use in experiment.'),
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
                     default='vdcnn9-maxpool', show_default=True,
                     help='Specify model architecture.'),
        click.option('--optimizer', '-o', type=click.Choice(['sgd', 'adam']),
                     default='sgd', show_default=True,
                     help='Specify optimizer for training.'),
        click.option('--epochs', '-e', multiple=True, type=int,
                     default=(3, 3, 3, 3, 3), show_default=True,
                     help='Specify epochs for training.'),
        click.option('learning_rates', '--learning-rate', '-l', multiple=True,
                     type=float,
                     default=(0.01, 0.005, 0.0025, 0.00125, 0.000625),
                     show_default=True,
                     help='Specify learning rate for training.'),
        click.option('--momentum', type=float, default=0.9, show_default=True,
                     help='Specify proxy momentum.'),
        click.option('--weight-decay', type=float, default=1e-4,
                     show_default=True,
                     help='Specify weight decay.'),
        click.option('--batch-size', '-b', default=128, show_default=True,
                     help='Specify minibatch size for training.'),
        click.option('--eval-batch-size', type=int,
                     callback=utils.override_option,
                     help='Override minibatch size for evaluation')
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

          datasets_dir: str, dataset: str,
          validation: int, shuffle: bool,

          arch: str, optimizer: str, epochs: Tuple[int, ...],
          learning_rates: Tuple[float, ...],
          momentum: float, weight_decay: float,
          batch_size: int, eval_batch_size: int,

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
@click.option('--initial-subset', type=int, default=72_000, show_default=True,
              help='Number of training examples to use for initial'
                   ' labelled set.')
@click.option('rounds', '--round', '-r', multiple=True, type=int,
              default=(288_000, 360_000, 360_000, 360_000, 360_000),
              show_default=True,
              help='Number of unlabelled examples to select in a round of'
                   ' labeling.')
@click.option('--selection-method', type=click.Choice(active_learning_methods),
              default='least_confidence', show_default=True,
              help='Criteria for selecting unlabelled examples to label')
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

           datasets_dir: str, dataset: str,
           validation: int, shuffle: bool,

           arch: str, optimizer: str,
           epochs: Tuple[int, ...],
           learning_rates: Tuple[float, ...],
           momentum: float, weight_decay: float,
           batch_size: int, eval_batch_size: int,

           proxy_arch: str, proxy_optimizer: str,
           proxy_epochs: Tuple[int, ...],
           proxy_learning_rates: Tuple[float, ...],
           proxy_momentum: float, proxy_weight_decay: float,
           proxy_batch_size: int, proxy_eval_batch_size: int,

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
@click.option('--subset', type=int, default=1_800_000,
              show_default=True,
              help='Number of examples to keep in the selected subset.')
@click.option('--selection-method', type=click.Choice(coreset_methods),
              default='least_confidence', show_default=True,
              help='Criteria for selecting examples')
@click.option('--precomputed-selection',
              help='Path to timestamp run_dir of precomputed indices')
@click.option('--train-target/--no-train-target',
              default=True, show_default=True,
              help=('If proxy and target are different, train the target'
                    ' after selection'))
@computing_options
@miscellaneous_options
def coreset(run_dir: str,

            datasets_dir: str, dataset: str,
            validation: int, shuffle: bool,

            arch: str, optimizer: str,
            epochs: Tuple[int, ...],
            learning_rates: Tuple[float, ...],
            momentum: float, weight_decay: float,
            batch_size: int, eval_batch_size: int,

            proxy_arch: str, proxy_optimizer: str,
            proxy_epochs: Tuple[int, ...],
            proxy_learning_rates: Tuple[float, ...],
            proxy_momentum: float, proxy_weight_decay: float,
            proxy_batch_size: int, proxy_eval_batch_size: int,

            subset: int, selection_method: str,
            precomputed_selection: Optional[str], train_target: bool,

            cuda: bool, device_ids: Tuple[int, ...],
            num_workers: int, eval_num_workers: int,

            seed: int, checkpoint: str, track_test_acc: bool):
    coreset_function(**locals())


@cli.command()
@click.argument('executable')
@click.option('--run-dir', default='./run', show_default=True,
              help='Path to log results and other artifacts.')
@click.option('--datasets-dir', default='./data', show_default=True,
              help='Path to datasets.')
@click.option('--dataset', '-d', type=click.Choice(DATASETS),
              default='amazon_review_polarity', show_default=True,
              help='Specify dataset to use in experiment.')
@click.option('--dim', default=10, show_default=True,
              help='Size of word vectors.')
@click.option('--ngrams', '-n', default=2, show_default=True,
              help='Max length of word ngram.')
@click.option('--min-count', default=1, show_default=True,
              help='Minimal number of word occurences.')
@click.option('--bucket', default=10_000_000, show_default=True,
              help='Number of buckets.')
@click.option('--learning-rate', default=0.05, show_default=True,
              help='Learning rate.')
@click.option('--epochs', default=5, show_default=True,
              help='Number of epochs to train.')
@click.option('sizes', '--size', multiple=True, type=int,
              default=(
                72_000, 360_000, 720_000, 1_080_000, 1_440_000, 1_800_000
              ),
              show_default=True,
              help=('Number of examples to keep after each selection round.'
                    ' The first number represents the initial subset.'
                    ' Increasing sizes represent the active learning use case.'
                    ' Decreasing sizes represent the core-set selection use'
                    ' case.'))
@click.option('--selection-method',
              type=click.Choice(FASTTEXT_SELECTION_METHODS),
              default='entropy', show_default=True,
              help='Criteria for selecting examples.')
@click.option('--threads', default=4, show_default=True,
              help='Number of threads.')
@click.option('--seed', '-s', type=int,
              help='Specify random seed.')
@click.option('--track-test-acc/--no-track-test-acc',
              default=True, show_default=True,
              help='Calculate performance of the models on the test '
                   ' data in addition or instead of the validation'
                   ' dataset.')
def fasttext(executable: str,

             run_dir: str, datasets_dir: str, dataset: str,

             dim: int, ngrams: int, min_count: int, bucket: int,
             learning_rate: float, epochs: int,

             sizes: Tuple[int, ...], selection_method: str,

             threads: int, seed: Optional[int], track_test_acc: bool):
    fasttext_function(**locals())


if __name__ == '__main__':
    if torch.cuda.is_available():
        cudnn.benchmark = True  # type: ignore
    cli()
