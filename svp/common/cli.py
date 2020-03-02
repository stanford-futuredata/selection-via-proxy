import torch
import click
from svp.common import utils


def computing_options(func):
    decorators = [
        click.option('--cuda/--no-cuda', default=True, show_default=True,
                     help="Enable or disable available GPUs"),
        click.option('device_ids', '--device', '-d', multiple=True, type=int,
                     default=tuple(range(torch.cuda.device_count())),
                     show_default=True,
                     help="Specify device ids for GPUs to use."),
        click.option('--num-workers', type=int, default=0, show_default=True,
                     help='Number of workers to use'
                          ' for data loading during training'),
        click.option('--eval-num-workers', type=int,
                     callback=utils.override_option,
                     help='Number of workers to use'
                          ' for data loading during evaluation')
    ]
    decorators.reverse()
    for decorator in decorators:
        func = decorator(func)
    return func


def miscellaneous_options(func):
    decorators = [
        click.option('--seed', '-s', type=int,
                     help='Specify random seed'),
        click.option('--checkpoint', '-c',
                     type=click.Choice(['best', 'all', 'last']),
                     default='best', show_default=True,
                     help='Specify when to create a checkpoint for the model:'
                          ' only the best performing model on the validation'
                          ' data ("best"), after every epoch ("all"), or'
                          ' only the last epoch of each segment of the'
                          ' learning rate schedule ("last").'),
        click.option('--track-test-acc/--no-track-test-acc',
                     default=True, show_default=True,
                     help='Calculate performance of the models on the test '
                          ' data in addition or instead of the validation'
                          ' dataset.')
    ]
    decorators.reverse()
    for decorator in decorators:
        func = decorator(func)
    return func
