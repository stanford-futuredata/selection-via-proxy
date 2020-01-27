import os
from glob import glob
from typing import Tuple, Optional
from functools import partial

import torch
import click
import numpy as np
import torch.backends.cudnn as cudnn

from svp.common import utils
from svp.cifar.models import MODELS
from svp.cifar.datasets import DATASETS, create_dataset
from svp.cifar.train import create_graph
from svp.common.active import (SELECTION_METHODS, PROXY_DIR_PREFIX,
                               TARGET_DIR_PREFIX, create_eval_loaders,
                               create_trainer, check_different_models,
                               select, validate_splits)


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
@click.option('--weight-decay', type=float, default=5e-4, show_default=True,
              help='Specify weight decay.')
@click.option('--batch-size', '-b', default=128, show_default=True,
              help='Specify minibatch size for training.')
@click.option('--eval-batch-size', type=int, callback=utils.override_option,
              help='Override minibatch size for evaluation')
# Proxy overrides
@click.option('--proxy-arch', '-a', type=click.Choice(MODELS.keys()),
              callback=utils.override_option,
              help='Override proxy model archtiecture.')
@click.option('--proxy-optimizer', '-o', type=click.Choice(['sgd', 'adam']),
              callback=utils.override_option,
              help='Specify optimizer for training.')
@click.option('--proxy-epochs', multiple=True, type=int,
              callback=utils.override_option,
              help='Override epochs for proxy training.')
@click.option('proxy_learning_rates', '--proxy-learning-rate',
              multiple=True, type=float,
              callback=utils.override_option,
              help='Override proxy learning rate for training.')
@click.option('--proxy-momentum', type=float, callback=utils.override_option,
              help='Override momentum.')
@click.option('--proxy-weight-decay', type=float,
              callback=utils.override_option,
              help='Override weight decay.')
@click.option('--proxy-batch-size', type=int, callback=utils.override_option,
              help='Override proxy minibatch size for training.')
@click.option('--proxy-eval-batch-size', type=int,
              callback=utils.override_option,
              help='Override proxy minibatch size for evaluation')
# Active learning options
@click.option('--initial-subset', type=int, default=1_000, show_default=True,
              help='Number of training examples to use for initial'
                   ' labelled set.')
@click.option('rounds', '--round', '-r', multiple=True, type=int,
              default=tuple(), show_default=True,
              help='Number of unlabelled examples to select in a round of'
                   ' labeling.')
@click.option('--selection-method', type=click.Choice(SELECTION_METHODS),
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
def active(run_dir: str,

           datasets_dir: str, dataset: str, augmentation: bool,
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
    validate_splits(train_dataset, validation, initial_subset, rounds)
    if augmentation:
        valid_dataset = create_dataset(dataset, datasets_dir, train=True,
                                       augmentation=False)
    else:
        valid_dataset = train_dataset
    test_dataset = create_dataset(dataset, datasets_dir, train=False,
                                  augmentation=False)
    assert len(train_dataset) == len(valid_dataset)
    num_classes = len(set(test_dataset.targets))  # type: ignore

    # Create data loaders
    train_indices, *loaders = create_eval_loaders(
        train_dataset, valid_dataset, test_dataset,
        validation=validation,
        shuffle=shuffle,
        run_dir=run_dir,
        target_eval_batch_size=eval_batch_size,
        proxy_eval_batch_size=proxy_eval_batch_size,
        num_workers=num_workers,
        use_cuda=use_cuda)
    proxy_valid_loader, proxy_test_loader, *loaders = loaders
    target_valid_loader, target_test_loader = loaders

    # Don't need an initial subset or proxy if the selections are given
    if precomputed_selection is None:
        proxy_partial = partial(create_graph,
                                arch=proxy_arch,
                                num_classes=num_classes,
                                optimizer=proxy_optimizer,
                                learning_rate=proxy_learning_rates[0],
                                momentum=proxy_momentum,
                                weight_decay=proxy_weight_decay)
        proxy_trainer = create_trainer(
            run_dir, num_classes, train_dataset,
            proxy_valid_loader, proxy_test_loader,
            create_graph=proxy_partial,
            epochs=proxy_epochs,
            learning_rates=proxy_learning_rates,
            batch_size=proxy_batch_size,
            num_workers=num_workers,
            device=device,
            device_ids=device_ids,
            checkpoint=checkpoint,
            use_cuda=use_cuda,
            track_test_acc=track_test_acc,
            prefix=PROXY_DIR_PREFIX)
        next(proxy_trainer)

    are_different_models = check_different_models(config)
    if train_target:
        if are_different_models or precomputed_selection is not None:
            target_partial = partial(create_graph,
                                     arch=arch,
                                     num_classes=num_classes,
                                     optimizer=optimizer,
                                     learning_rate=learning_rates[0],
                                     momentum=momentum,
                                     weight_decay=weight_decay)
            target_trainer = create_trainer(
                run_dir, num_classes, train_dataset,
                target_valid_loader, target_test_loader,
                create_graph=target_partial,
                epochs=epochs,
                learning_rates=learning_rates,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                device_ids=device_ids,
                checkpoint=checkpoint,
                use_cuda=use_cuda,
                track_test_acc=track_test_acc,
                prefix=TARGET_DIR_PREFIX)
            next(target_trainer)
        else:
            print('Proxy and target are not different.')
            proxy_dir = os.path.join(run_dir, PROXY_DIR_PREFIX)
            target_dir = os.path.join(run_dir, TARGET_DIR_PREFIX)
            os.symlink(os.path.relpath(proxy_dir,
                                       os.path.dirname(target_dir)),
                       target_dir)
            print(f'Linked {target_dir} to {proxy_dir}')

    if precomputed_selection is not None:
        assert train_target, "Must train target if selection is precomuted"
        assert os.path.exists(precomputed_selection)

        # For convenience
        selector_csv = os.path.join(precomputed_selection, 'selector.csv')
        os.symlink(os.path.relpath(selector_csv, run_dir),
                   os.path.join(run_dir, 'selector.csv'))

        selection_csv = os.path.join(precomputed_selection, 'selection.csv')
        os.symlink(os.path.relpath(selection_csv, run_dir),
                   os.path.join(run_dir, 'selection.csv'))

        files = glob(os.path.join(precomputed_selection, 'selector',
                     '*', 'labelled_*.index'))
        indices = [np.loadtxt(file, dtype=np.int64) for file in files]
        selections = sorted(
            zip(files, indices),
            key=lambda selection: len(selection[1]))  # type: ignore

        for file, labelled in selections:
            print('Load labelled indices from {}'.format(file))
            should_eval = (len(eval_target_at) == 0 or
                           len(labelled) in eval_target_at)
            if should_eval:
                _, stats = target_trainer.send(labelled)
                utils.save_result(stats, os.path.join(run_dir, "target.csv"))
    else:
        # Create initial random subset
        num_train = len(train_indices)
        labelled = np.random.permutation(train_indices)[:initial_subset]
        utils.save_index(labelled, run_dir,
                         'initial_subset_{}.index'.format(len(labelled)))

        model, stats = proxy_trainer.send(labelled)
        utils.save_result(stats, os.path.join(run_dir, "selector.csv"))

        for selection_size in rounds:
            labelled, stats = select(model, train_dataset,
                                     current=labelled,
                                     pool=train_indices,
                                     budget=selection_size,
                                     method=selection_method,
                                     batch_size=proxy_batch_size,
                                     device=device,
                                     device_ids=device_ids,
                                     num_workers=num_workers,
                                     use_cuda=use_cuda)
            utils.save_result(stats, os.path.join(run_dir, 'selection.csv'))

            model, stats = proxy_trainer.send(labelled)
            utils.save_result(stats, os.path.join(run_dir, "selector.csv"))
            should_eval = (len(eval_target_at) == 0 or
                           len(labelled) in eval_target_at)
            if train_target and should_eval and are_different_models:
                _, stats = target_trainer.send(labelled)
                utils.save_result(stats, os.path.join(run_dir, "target.csv"))


if __name__ == '__main__':
    if torch.cuda.is_available():
        cudnn.benchmark = True  # type: ignore
    active()
