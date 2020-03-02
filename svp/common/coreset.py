import numpy as np
from torch.utils.data import Dataset

from svp.common.selection import UNCERTAINTY_METHODS


PROXY_DIR_PREFIX = 'selector'
TARGET_DIR_PREFIX = 'target'
SELECTION_METHODS = ['kcenters', 'random', 'forgetting_events']
SELECTION_METHODS += UNCERTAINTY_METHODS


def validate_splits(train_dataset: Dataset, validation: int, subset: int):
    """
    Check there is enough training data for validation and the subset.

    Parameters
    ----------
    train_dataset : Dataset
        Dataset for training.
    validation : int
        Number of examples to use for validation.
    subset : int
        Number of example to use for selected subset.
    """
    num_train = len(train_dataset)
    assert validation < num_train and validation >= 0

    num_train -= validation
    assert subset <= num_train and subset > 0


class ForgettingEventsMeter:
    def __init__(self, dataset):
        self.correct = np.zeros(len(dataset), dtype=np.int64)
        self.nevents = np.zeros(len(dataset), dtype=np.float64)
        self.was_correct = np.zeros(len(dataset), dtype=np.bool)

    def callback(self, indices, inputs, targets, outputs):
        # TODO: there is a better way to do this
        correct_batch = targets.eq(outputs.argmax(dim=1)).cpu().numpy().astype(np.int64)  # noqa: E501
        transitions = correct_batch - self.correct[indices]

        self.correct[indices] = correct_batch
        self.was_correct[indices] |= correct_batch.astype(np.bool)
        self.nevents[indices[transitions == -1]] += 1.
