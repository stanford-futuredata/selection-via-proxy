import re
import os
import subprocess
from datetime import datetime
from contextlib import contextmanager
from collections import OrderedDict
from typing import Tuple, Optional, Mapping, Dict, Any

import numpy as np
import pandas as pd

from svp.common import utils
from svp.common.selection import UNCERTAINTY_METHODS

# Using shell command to match original example for fastText
NORMALIZE_TEXT = """
cat {input} | tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \\
sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\\./ \\. /g' -e 's/<br \\/>/ /g' \\
    -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\\!/ \\! /g' \\
    -e 's/\\?/ \\? /g' -e 's/\\;/ /g' -e 's/\\:/ /g' | tr -s " " > {output}
"""

FASTTEXT_TRAIN = """
{command} supervised -input "{train_file}" -output "{model_file}" -dim {dim} \
-lr {lr} -wordNgrams {ngrams}  -minCount {min_count} -bucket {bucket} \
-epoch {epoch} -thread {threads}
"""

FASTTEXT_TEST = """
{command} test "{model_bin}" "{test_file}"
"""

FASTTEXT_PREDICT = """
{command} predict "{model_bin}" "{test_file}" > "{test_predict}"
"""

# Only want to rank training data points
FASTTEXT_PROBS = """
{command} predict-prob "{model_bin}" "{full_train_file}" {num_classes} \
> "{train_probs}"
"""

LABEL_PATTERN = re.compile(r'^__label__(?P<label>\d+)')
FASTTEXT_SELECTION_METHODS = ['random']
FASTTEXT_SELECTION_METHODS += UNCERTAINTY_METHODS


def fasttext(executable: str,

             run_dir: str = './run',

             datasets_dir: str = './data',
             dataset: str = 'amazon_review_polarity',

             dim: int = 10, ngrams: int = 2, min_count: int = 1,
             bucket: int = 10_000_000, learning_rate: float = 0.05,
             epochs: int = 5,

             sizes: Tuple[int, ...] = (
                72_000, 360_000, 720_000, 1_080_000, 1_440_000, 1_800_000
             ),
             selection_method: str = 'entropy',

             threads: int = 4, seed: Optional[int] = None,
             track_test_acc: bool = True):
    """
    Perform active learning or core-set selection with fastText.

    Training and evaluation are both performed using fastText v0.1.0. Please
    download and install fastText before using this function:

    https://github.com/facebookresearch/fastText/tree/v0.1.0

    Active learning is represented by increasing sizes. To make up the
    difference between sizes, additional examples are selected according to
    the `selection_method` from remaining pool of data. For example,
    `sizes = [100, 200, 300, 400]` would start with a subset of 100 randomly
    selected examples and select a series of 3 batches of 100-examples each.

    Core-set selection is represented by decreasing sizes. Examples are
    selected according to the `selection_method` from entire pool of data.
    This assumes that the first size is the size of the entire labeled data,
    but it is not explicitly enforced. For example, to reproduce the
    "selection via proxy" core-set experiments on Amazon Review Polarity,
    `sizes` should be length 2 with `sizes[0] == 3_600_000` and `sizes[1]`
    set to the desired subset sizes.

    Parameters
    ----------
    executable : str
        Path to fastText executable. Please install fastText v0.1.0:
        https://github.com/facebookresearch/fastText/tree/v0.1.0
    run_dir : str, default './run'
        Path to log results and other artifacts.
    datasets_dir : str, default './data'
        Path to datasets.
    dataset : str, default 'amazon_review_polarity'
        Dataset to use in experiment (e.g., amazon_review_full)
    dim : int, default 10
        Size of word vectors.
    ngrams : int, default 2
        Max length of word ngram.
    min_count : int, default 1
        Minimal number of word occurences.
    bucket : int, default 10_000_000
        Number of buckets.
    learning_rate : float, default 0.05
        Learning rate.
    epochs : int, default 5
        Number of epochs to train.
    sizes : Tuple[int, ...], default (
            72,000, 360,000, 720,000, 1,080,000, 1,440,000, 1,800,000)
        Number of examples to keep after each selection round. The first number
        represents the initial subset. Increasing sizes represent the active
        learning use case. Decreasing sizes represent the core-set selection
        use case.
    selection_method : str, default 'entropy'
        Criteria for selecting examples.
    threads : int, default 4
        Number of threads.
    seed : Optional[int], default None
        Random seed for numpy, torch, and others. If None, a random int is
        chosen and logged in the experiments config file.
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

    # Create proxy directory to be compatible with other scripts.
    proxy_run_dir = os.path.join(run_dir, 'proxy')
    os.makedirs(proxy_run_dir, exist_ok=True)

    # Create the training dataset, which is just the path to the
    #   normalized training text file. This allows us to work with the
    #   fasttext executable.
    train_dataset = create_dataset(dataset, datasets_dir, train=True)

    # Create train indices
    with open(train_dataset) as file:
        N = sum(1 for line in file)

    # Validate sizes
    _sizes = np.array(sizes)
    assert len(_sizes), 'Need at least two sizes'
    assert (_sizes <= N).all(), f'{N} is insufficient sizes {_sizes}'
    assert (_sizes > 0).all(), f'Sizes must be positive. Got {_sizes}'
    changes = _sizes[1:] - _sizes[:-1]
    assert (changes != 0).all(), f'Sizes must change'
    signs = np.sign(changes)
    assert (signs[1:] == signs[0]).all(), f'Sizes must increase or decrease monotonically'  # noqa: E501
    if signs[0] > 0:
        # Positive signs mean active learning (i.e., add labeled examples)
        print(f'Valid active learning experiment: {sizes}')
        setting = 'active'
    else:
        # Negative signs mean core-set selection (i.e., remove examples)
        print(f'Valid core-set selection experiment: {sizes}')
        setting = 'coreset'

    print('{} total examples for training'.format(N))

    # Read training labels and calculate the number of classes.
    targets = read_labels(train_dataset)
    num_classes = len(set(targets))

    test_dataset = None
    if track_test_acc:
        # Create the test dataset, which is just the path to the
        #   normalized test text file. This allows us to work with the
        #   fasttext executable.
        test_dataset = create_dataset(dataset, datasets_dir, train=False)

    # Create initial subset to train fastText.
    initial_subset = sizes[0]
    print('Selecting initial subset of {}'.format(initial_subset))
    # Keep a random subset
    labeled = np.random.permutation(np.arange(N))[:initial_subset]

    # Pulling from the normalized training data, save a separate file
    #   with only the examples in `labeled`. This allows us to work
    #   with the fasttext executable
    tag = f'{initial_subset}'
    round_dir = os.path.join(proxy_run_dir, tag)
    os.makedirs(round_dir, exist_ok=True)
    current_path = os.path.join(round_dir, f'shuffled_{tag}.norm')
    _ = shuffle_and_index(train_dataset, current_path, kept=labeled)
    utils.save_index(labeled, run_dir,
                     'initial_subset_{}.index'.format(len(labeled)))

    # Perform selection(s)
    print('Running active learning for budgets: {}'.format(sizes[1:]))
    # Skip the first selection, which was performed above.
    for next_size in sizes[1:]:
        # Train fastText on the labeled data we have so far.
        print("Training on {} examples".format(tag))
        probs, train_stats, selection_stats = train_fastText(
            executable, current_path, train_dataset, test_dataset,
            round_dir, tag, num_classes,
            lr=learning_rate,
            dim=dim,
            min_count=min_count,
            bucket=bucket,
            epoch=epochs,
            threads=threads,
            ngrams=ngrams,
            verbose=True)
        utils.save_result(train_stats, os.path.join(run_dir, "proxy.csv"))

        print('Selecting examples for size {}'.format(next_size))
        # Rank examples based on the probabilities from fastText.
        ranking_start = datetime.now()
        ranking = calculate_rank(probs, selection_method)

        # Select top examples.
        if next_size > len(labeled):
            # Performing active learning.
            # Add top ranking examples to the existing labeled set.
            labeled_set = set(labeled)
            ranking = [i for i in ranking if i not in labeled_set]
            new_indices = ranking[:(next_size - len(labeled))]
            selection_stats['current_nexamples'] = len(labeled)
            selection_stats['new_nexamples'] = len(new_indices)
            labeled = np.concatenate([labeled, new_indices])
        elif next_size < len(labeled):
            # Performing core-set selection.
            # `labeled` should include the entire training set, so
            #   this is taking a strict subset. This is not
            #   explicitly enforced to allow for additional
            #   exploration and experimentation (e.g., what if the
            #   proxy only looked at a subset of the data before
            #   selecting the best examples from all the data?).
            labeled = ranking[:next_size]
            selection_stats['current_nexamples'] = 0
            selection_stats['new_nexamples'] = len(labeled)
        ranking_time = datetime.now() - ranking_start
        assert len(set(labeled)) == next_size

        # Pulling from the normalized training data, save a separate file
        #   with only the examples in `labeled`. This allows us to work
        #   with the fasttext executable
        tag = f'{next_size}'
        round_dir = os.path.join(proxy_run_dir, tag)
        os.makedirs(round_dir, exist_ok=True)
        current_path = os.path.join(round_dir, f'shuffled_{tag}.norm')
        _ = shuffle_and_index(train_dataset, current_path, kept=labeled)
        utils.save_index(labeled, round_dir,
                         'labeled_{}.index'.format(len(labeled)))

        # Save high-level runtimes for analysis
        selection_stats['ranking_time'] = ranking_time
        selection_stats['total_time'] = selection_stats['inference_time']
        selection_stats['total_time'] += selection_stats['ranking_time']
        utils.save_result(selection_stats,
                          os.path.join(run_dir, 'selection.csv'))

    if setting == 'coreset':
        # Resave indices to be compatible with core-set experiments
        utils.save_index(labeled, run_dir, 'selected.index')
        utils.save_index(np.array([], dtype=np.int64), run_dir, 'dev.index')


def create_dataset(dataset: str, dataset_dir: str, train: bool = True) -> str:
    """
    Create path to normalized fastText data.

    Parameters
    ----------
    dataset : str
        Dataset to use in experiment (e.g., amazon_review_full)
    datasets_dir : str
        Path to datasets.
    train: bool, default True
        Load training data.

    Returns
    -------
    normalized_data_path : str
    """
    dataset_dir = os.path.join(dataset_dir, dataset + '_csv')

    mode = 'train' if train else 'test'
    # Normalize data for fastText
    filename = 'normalized.'
    filename += mode
    # fastText needs to preprocess data before it can train
    normalized_data_path = os.path.join(dataset_dir, filename)
    if not os.path.exists(normalized_data_path):
        # Preprocessed data doesn't exist, so we need to create it from
        #   the original data.
        print(f"Normalize {mode} data doesn't exist. Going to raw data.")
        raw_filename = mode
        raw_filename += '.csv'
        raw_path = os.path.join(dataset_dir, raw_filename)
        assert os.path.exists(raw_path), f"Raw {mode} data doesn't exist: {raw_path}"  # noqa: E501
        with runtime(f"Normalized raw {mode} data from {raw_path}"):
            normalize_text(raw_path, normalized_data_path)
    print(f'Using {normalized_data_path} for {mode} data')
    return normalized_data_path


@contextmanager
def runtime(text: str):
    """
    Calculate wall-clock time for anything run within this context.

    Parameters
    ----------
    text : str
        Output text while context is running.
    """
    print(text, end="")
    start = datetime.now()
    try:
        yield start
    finally:
        print(" (finished in {})".format(datetime.now() - start))


def normalize_text(input: str, output: str):
    """
    Normalize data (examples + labels) according to fastText rules.

    Parameters
    ----------
    input : str
        Path to raw data.
    output : str
        Path to output normalized data.
    """
    run_command(NORMALIZE_TEXT.format(input=input, output=output))


def shuffle_and_index(prev: str, curr: str, kept: np.array,
                      mapping: Optional[Mapping[int, int]] = None,
                      shuffle: bool = True) -> Tuple[np.array, np.array]:
    """
    Shuffle lines (i.e., examples) in a text file.

    Parameters
    ----------
    prev : str
        Path to raw data to shuffle.
    curr : str
        Path to output shuffled data.
    kept : np.array of ints
        Line numbers to keep from `prev` text file.
    mapping : Optional[Mapping[int, int]], default None
        Optionally mapping for indices.
    shuffle : bool, default True
       Shuffle the line numbers to keep.

    Returns
    -------
    indices : np.array
    targets : np.array
    """
    with open(prev) as file:
        lines = file.readlines()

    if shuffle:
        kept = np.random.permutation(kept)

    indices = np.zeros(len(kept), dtype=np.int64)
    targets = np.zeros(len(kept), dtype=np.int64)
    with open(curr, 'w') as file:
        for index, example in enumerate(kept):
            line = lines[example]
            match = re.match(LABEL_PATTERN, line)
            assert match is not None, "Every line should have a label"
            targets[index] = int(match.group(1)) - 1
            indices[index] = example if mapping is None else mapping[example]
            file.write(line)

    pd.Series(indices).to_csv(curr + '.index')
    return indices, targets


def train_fastText(command: str, train_file:  str, full_train_file: str,
                   test_file: Optional[str], run_dir: str, tag: str,
                   num_classes: int,
                   dim: int = 10, lr: float = 0.05, ngrams: int = 2,
                   epoch: int = 5, threads: int = 4, min_count: int = 1,
                   bucket: int = 10000000, verbose=True
                   ) -> Tuple[np.array, Dict[str, Any], Dict[str, Any]]:
    """
    Train fastText model.

    Parameters
    ----------
    command : str
        Path to fasttext executable.
    train_file : str
        Path to fastText normalized data to use for training.
    full_train_file : str
        Path to fastText normalized data to calculate probabilities for.
    test_file : Optional[str]
        Path to fastText normalized data to use for evaluation.
    run_dir : str, default './run'
        Path to log results and other artifacts.
    tag : str
        Unique identifier to add to the filenames for the model binary, test
        predictions, and probabilities.
    num_classes : int
        Number of classes for the output probabilities.
    dim : int, default 10
        Size of word vectors.
    learning_rate : float, default 0.05
        Learning rate.
    ngrams : int, default 2
        Max length of word ngram.
    epochs : int, default 5
        Number of epochs to train.
    threads : int, default 4
        Number of threads.
    min_count : int, default 1
        Minimal number of word occurences.
    bucket : int, default 10_000_000
        Number of buckets.
    verbose : bool
        Output the command lines used with the fastText executable before
        execution.

    Returns
    -------
    probs : np.array
    train_stats : Dict[str, Any]
    selection_stats : Dict[str, Any]
    """
    model_file = os.path.join(run_dir, f"model_{tag}")  # noqa: F841
    model_bin = model_file + '.bin'  # noqa: F841
    model_vec = model_file + '.vec'  # noqa: F841
    test_predict = os.path.join(run_dir, f"test_predict_{tag}.test")  # noqa: F841 E501

    # Only want probabilities for training data for ranking
    train_probs = os.path.join(run_dir, f"train_probs_{tag}.train")
    config = {k: v for k, v in locals().items()}

    train_stats: Dict[str, Any] = OrderedDict()
    selection_stats: Dict[str, Any] = OrderedDict()
    with open(train_file) as file:
        nlabeled = sum(1 for line in file)
    train_stats['nexamples'] = nlabeled

    # train
    train_start = datetime.now()
    command = FASTTEXT_TRAIN.format(**config)
    run_command(command, verbose=verbose)
    train_time = datetime.now() - train_start
    train_stats['train_time'] = train_time

    inference_start = datetime.now()
    command = FASTTEXT_PROBS.format(**config)
    run_command(command, verbose=verbose)
    probs = read_fasttext_probs(train_probs, num_classes=num_classes)
    inference_time = datetime.now() - inference_start
    selection_stats['inference_time'] = inference_time

    train_acc = calculate_train_accuracy(full_train_file, probs)
    train_stats['train_accuracy'] = train_acc
    print(f'Train accuracy: {train_acc}')

    if test_file is not None:
        # predict test
        test_start = datetime.now()
        command = FASTTEXT_PREDICT.format(**config)
        run_command(command, verbose=verbose)
        test_acc = calculate_test_accuracy(config['test_file'], test_predict)
        print(f'Test accuracy: {test_acc}')
        test_time = datetime.now() - test_start
        train_stats['test_time'] = test_time
        train_stats['test_accuracy'] = test_acc

    return (probs, train_stats, selection_stats)


def read_labels(path: str) -> np.array:
    """
    Read labels from fastText normalized files.

    Parameters
    ----------
    path : str
        Path to fastText normalized file.

    Returns
    -------
    labels : np.array
    """
    labels = []
    with open(path) as file:
        for line in file:
            match = re.match(LABEL_PATTERN, line)
            assert match is not None
            label = int(match.group('label'))
            labels.append(label)
    return np.array(labels, dtype=np.int64) + 1


def calculate_train_accuracy(targets_path: str, probs: np.array) -> float:
    """
    Calculate accuracy on training data.

    Parameters
    ----------
    targets_path : str
        Path to fastText normalized training data.
    probs : np.array
        Class probability distribution for each line the `targets_path`

    Returns
    -------
    accuracy : float
    """
    targets = read_labels(targets_path)
    preds = probs.argmax(axis=1)
    assert len(targets) == len(preds)
    return (targets == preds).sum() / len(targets)


def calculate_test_accuracy(targets_path: str, preds_path: str) -> float:
    """
    Calculate accuracy on test data.

    Parameters
    ----------
    targets_path : str
        Path to fastText normalized test data.
    preds_path : str
        Path to fastText normalized predicted labels.

    Returns
    -------
    accuracy : float
    """
    targets = read_labels(targets_path)
    preds = read_labels(preds_path)
    assert len(targets) == len(preds)
    return (targets == preds).sum() / len(targets)


def calculate_rank(probs, rank_metric) -> np.array:
    """
    Rank examples based on class probabilites.

    Parameters
    ----------
    probs : np.array
        Class probability distribution for each line the `targets_path`
    rank_metric : str
        Criteria for selecting examples.

    Returns
    -------
    ranking : np.array
    """
    if rank_metric == 'entropy':
        entropy = (np.log(probs) * probs).sum(axis=1) * -1.
        ranking = entropy.argsort()[::-1]
    elif rank_metric == 'least_confidence':
        probs = probs.max(axis=1)
        ranking = probs.argsort(axis=0)
    elif rank_metric == 'random':
        N = len(probs)
        ranking = np.arange(N)
        np.random.shuffle(ranking)
    else:
        raise NotImplementedError(f"{rank_metric} hasn't been implemented yet")
    return ranking


def read_fasttext_probs(path: str,
                        num_classes: Optional[int] = None) -> np.array:
    '''
    Read probabilies from fastText's predict-prob command.

    Parameters
    ----------
    path : str
        Path to output file from fastText's predict-prob command.
    num_classes : Optional[int], default None
        Specify the number of classes in the output probability
        distribution. If None, this is inferred from the first line of
        the file.

    Returns
    -------
    probs : np.array
    '''
    records = []
    with open(path, 'r') as file:
        for line in file:
            split = line.strip().replace('__label__', '').split(' ')
            labels = map(lambda x: int(x) - 1, split[::2])
            values = map(float, split[1::2])
            record = {label: value for label, value in zip(labels, values)}
            records.append(record)
    if num_classes is None:
        num_classes = len(records[0].keys())
    df = pd.DataFrame.from_records(records).loc[:, np.arange(num_classes)]
    return df.values


def run_command(command, shell=True, verbose=False, required=True):
    """
    Run command as subprocess
    """
    if verbose:
        print(f"Running: {command}")
    result = subprocess.call(command, shell=shell)
    if required:
        assert result == 0, f"Failed command: {command}"
