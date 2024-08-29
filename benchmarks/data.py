# Copyright 2024 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the “License”);
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/cavi-cmn/blob/main/license.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import (
    Iterator,
    Iterable,
    Optional,
    Sequence,
    List,
    TypeVar,
    Generic,
    Sized,
    Union,
    Tuple,
)
import numpy as np

import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold

from jax import nn
import jax.numpy as jnp
from jax import random as jr
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

from jax.numpy import expand_dims as expand

import os
from ucimlrepo import fetch_ucirepo
from collections import defaultdict, Counter

import subprocess
import pandas as pd


ROOT_DIRECTORY = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly.

    Args:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized

    def __init__(
        self, key, data_source: Sized, num_samples: Optional[int] = None, generator=None
    ) -> None:
        self.key = key
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = jr.randint(
                key=self.key, shape=(1,), minval=0, maxval=2**15, dtype=jnp.int32
            ).item()
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        for _ in range(self.num_samples // n):
            yield from torch.randperm(n, generator=generator).tolist()

        yield from torch.randperm(n, generator=generator).tolist()[
            : self.num_samples % n
        ]

    def __len__(self) -> int:
        return self.num_samples


class JaxNumpyLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=jnp_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def jnp_collate(batch):
    return tree_map(jnp.array, default_collate(batch))


class ToNumpyArray(object):
    def __call__(self, x):
        x = np.array(x, dtype=jnp.float32)
        return x[None, ...] if x.ndim == 2 else x


def normalizeData(X):
    """Normalize the data to have zero mean and unit variance"""
    X_mean, X_std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    return (X - X_mean) / X_std


class ToTensor(object):
    def __call__(self, x):
        x = np.array(x, dtype=np.float32)
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(0)
        return x


class Flatten(object):
    def __call__(self, x):
        return torch.ravel(x)


class BaseDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        Xn = self.X[idx]
        yn = self.y[idx]
        if self.transform:
            Xn = self.transform(Xn)
        if self.target_transform:
            yn = self.target_transform(yn)
        return Xn, yn


def split_raw_data(X, y, train_size, test_size, max_train_size=400, random_state=0):
    """
    split the dataset into train / test split using. We first specify a value of max_train_size
    because we want to have the same test split when using different train sizes. This is useful
    because we want to compare the performance of different models trained on differently-sized training sets,
    but tested on the same test set.
    """
    assert (
        train_size <= max_train_size
    ), f"train size: {train_size:.0f} > max_train_size: {max_train_size:.0f}. Try to reduce the max_train_size."

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=max_train_size,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]

    return X_train, y_train, X_test, y_test


def create_balanced_train_test_split(
    X, Y, min_test_size=150, max_train_imbalance=0.05, max_test_imbalance=0.05
):
    """
    Creates balanced train and test sets from the input data while ensuring that the proportion of each class in the
    sets does not deviate significantly from the desired balance.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
      The feature matrix containing the samples.
    - Y: array-like, shape (n_samples,)
      The label vector containing the class labels.
    - min_test_size: int, optional (default=150)
      The minimum number of samples to be included in the test set across all classes.
    - max_train_imbalance: float, optional (default=0.05)
      The maximum allowable deviation from perfect balance in the training set, expressed as a proportion of the ideal
      class distribution (1/number of classes).
    - max_test_imbalance: float, optional (default=0.05)
      The maximum allowable deviation from perfect balance in the test set, expressed as a proportion of the ideal
      class distribution (1/number of classes).

    Returns:
    - X_train: array-like, shape (n_train_samples, n_features)
      The feature matrix for the training set.
    - X_test: array-like, shape (n_test_samples, n_features)
      The feature matrix for the test set.
    - Y_train: array-like, shape (n_train_samples,)
      The label vector for the training set.
    - Y_test: array-like, shape (n_test_samples,)
      The label vector for the test set.

    The function first ensures a minimum number of test samples per class based on the `min_test_size` parameter.
    It then iteratively adds additional samples to both the training and test sets, while maintaining the class
    proportions within the specified tolerance limits (`max_train_imbalance` and `max_test_imbalance`).
    """
    # Get unique classes and their counts
    class_counts = Counter(Y)
    classes = list(class_counts.keys())
    min_class_count = min(class_counts.values())

    # Determine the number of samples per class for test set
    n_classes = len(classes)
    samples_per_class_test = min(min_test_size // n_classes, min_class_count)
    samples_per_class_train = min_class_count - samples_per_class_test
    expected_proportion = 1 / n_classes

    # Initialize lists to hold the balanced train and test samples and used indices
    X_train, X_test, Y_train, Y_test = [], [], [], []
    used_train_indices = {}
    used_test_indices = {}

    for cls_i in classes:
        # Extract samples for this class
        cls_indices = np.where(Y == cls_i)[0]
        cls_samples_X = X[cls_indices]
        cls_samples_Y = Y[cls_indices]

        # Shuffle the samples
        indices = np.arange(len(cls_indices))
        np.random.shuffle(indices)

        # Split the samples into train and test sets
        cls_test_indices = indices[:samples_per_class_test]
        cls_train_indices = indices[
            samples_per_class_test : (samples_per_class_test + samples_per_class_train)
        ]

        # Save used indices for later reference
        used_train_indices[cls_i] = cls_train_indices
        used_test_indices[cls_i] = cls_test_indices

        X_train.extend(cls_samples_X[cls_train_indices])
        Y_train.extend(cls_samples_Y[cls_train_indices])
        X_test.extend(cls_samples_X[cls_test_indices])
        Y_test.extend(cls_samples_Y[cls_test_indices])

    # Attempt to add more samples while keeping the imbalance within the tolerance
    for cls_i in classes:
        # Current indices for this class
        cls_indices = np.where(Y == cls_i)[0]
        cls_samples_X = X[cls_indices]
        cls_samples_Y = Y[cls_indices]

        # Shuffle the samples
        indices = np.arange(len(cls_indices))
        np.random.shuffle(indices)

        # Determine the remaining samples not used in the initial split
        initial_used_indices = np.concatenate(
            (used_test_indices[cls_i], used_train_indices[cls_i])
        )
        remaining_indices = np.setdiff1d(indices, initial_used_indices)

        for idx in remaining_indices:
            # Calculate the proportion of this class in train and test sets if this sample is added
            new_train_count = Y_train.count(cls_i) + 1
            new_test_count = Y_test.count(cls_i) + 1

            train_proportion = new_train_count / len(Y_train)
            test_proportion = new_test_count / len(Y_test)

            train_tolerance_check = (
                abs(train_proportion - expected_proportion) <= max_train_imbalance
            )
            test_tolerance_check = (
                abs(test_proportion - expected_proportion) <= max_test_imbalance
            )

            # Add the new sample to the train or test set, updating proportions for the next iteration
            if train_tolerance_check:
                X_train.append(cls_samples_X[idx])
                Y_train.append(cls_samples_Y[idx])
            if test_tolerance_check:
                X_test.append(cls_samples_X[idx])
                Y_test.append(cls_samples_Y[idx])

    # Convert lists to arrays
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, X_test, Y_train, Y_test


def split_raw_data_old(X, y, train_size, test_size=200):
    """
    Using this function, we assume never do K-fold validation. To achieve a reasonble test
    size (150-250), we relax the balancing in test size, i.e. by adding the samples that
    are trimmed off via balancing process into the test size. Technically this is jusitifiable,
    since other datasets such as MNIST also contains slight imbalance between classes.

    argument:
        train_size: number of training samples we want
        max_train_size: number of samples we use to subsample training samples. One can think of
        this argument as a dividing index for training samples and test samples, i.e.
        X_train = X_total[:max_train_size][:train_size]
        X_test = X_total[max_train_size:]
        This makes sure that we always use the same test set for different train sets.
    """
    """
    test_size
    """
    # class labels
    labels = np.unique(y)
    num_classes = labels.shape[0]
    total_size = y.shape[0]

    # compute the number of per-class samples in the smallest class
    min_total_size_per_class = float("inf")
    for l in labels:
        min_total_size_per_class = min(min_total_size_per_class, (y == l).sum())
    # To max out the number of samples with evenly distributed class labels,
    # the smallest test size and largest train size
    min_test_size = total_size - num_classes * min_total_size_per_class
    max_train_size = num_classes * min_total_size_per_class

    if test_size < min_test_size:
        print(
            f"Automatically increase the test size from expected value: {test_size} to the smallest avaiable size : {min_test_size}"
        )
        test_size = min_test_size

    extra_test_size = test_size - min_test_size
    extra_test_size_per_class = extra_test_size // num_classes
    extra_test_size_rounded = extra_test_size_per_class * num_classes
    rounding_test = extra_test_size - extra_test_size_rounded

    if (min_total_size_per_class - 1) < extra_test_size_per_class:
        raise ValueError(
            f"Expected / Maximum test size: {test_size:.0f} / {(min_total_size_per_class - 1)*num_classes+min_test_size:.0f}."
        )

    if rounding_test > 0:
        print(
            f"To make evenly distributed train set, round down the test size from {test_size} to {extra_test_size_rounded+min_test_size}"
        )
        extra_test_size = extra_test_size_rounded

    # Given that test size, the max train size
    max_train_size_per_class = min_total_size_per_class - extra_test_size_per_class
    max_train_size = max_train_size_per_class * num_classes

    if train_size > max_train_size:
        possible_train_size = num_classes * min_total_size_per_class
        if train_size <= possible_train_size:
            possible_test_size = total_size - train_size

            raise ValueError(
                f"Expected / Maximum train size: {train_size:.0f} / {max_train_size:.0f}. Either reduce test size to {possible_test_size:.0f} to maintain this train size, or reduce train size under {possible_train_size:.0f}."
            )
        else:
            raise ValueError(
                f"Expected / Maximum train size: {train_size:.0f} / {max_train_size:.0f}. Reduce train size under {possible_train_size:.0f}."
            )

    train_size_per_class = train_size // num_classes
    train_size_rounded = train_size_per_class * num_classes

    if (train_size - train_size_rounded) > 0:
        print(
            f"To make evenly distributed train set, round down the train size from {train_size} to {train_size_rounded}"
        )
        train_size = train_size_rounded

    X_train, y_train, X_test, y_test = [], [], [], []
    for l in labels:
        idx_class_l = np.where(y == l)[0]
        X_class_l = X[idx_class_l]
        y_class_l = y[idx_class_l]
        X_train.append(X_class_l[:max_train_size_per_class][:train_size_per_class])
        y_train.append(y_class_l[:max_train_size_per_class][:train_size_per_class])
        X_test.append(X_class_l[max_train_size_per_class:])
        y_test.append(y_class_l[max_train_size_per_class:])
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    return X_train, y_train, X_test, y_test


def create_two_moons_generator(
    key,
    batch_size,
    num_training_samples,
    num_test_samples,
    num_standardization_samples=1e4,
    data_noise=5e-2,
):
    # Generate a large amount of data to compute statistics for standardizing the training and test data
    key, train_data_key, test_data_key = jr.split(key, 3)
    seeds = jr.randint(key, shape=(2,), minval=0, maxval=2**15, dtype=jnp.int32)
    X, _ = make_moons(
        n_samples=int(num_standardization_samples),
        noise=data_noise,
        random_state=seeds[0].item(),
    )

    X_mean, X_std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)

    # Generate the two moons dataset (both training and test sets)
    X, Y = make_moons(
        n_samples=num_training_samples + num_test_samples,
        noise=data_noise,
        random_state=seeds[1].item(),
    )

    X = (
        X - X_mean
    ) / X_std  # standardize the full (training and test) data using the statistics computed above

    X_train, Y_train = X[:num_training_samples], Y[:num_training_samples]
    Dataset = BaseDataset(X_train, Y_train, transform=None)
    train_dataloader = JaxNumpyLoader(
        Dataset,
        batch_size=batch_size,
        sampler=RandomSampler(key=train_data_key, data_source=Dataset),
    )

    X_test, Y_test = X[-num_test_samples:], Y[-num_test_samples:]
    Dataset = BaseDataset(X_test, Y_test, transform=None)
    test_dataloader = JaxNumpyLoader(
        Dataset,
        batch_size=num_test_samples,
        sampler=RandomSampler(key=test_data_key, data_source=Dataset),
    )

    return train_dataloader, test_dataloader


def create_iris_generator(key, num_training_samples, batch_size, num_folds=5):
    """
    Create a generator for the Iris dataset using the Stratified K-Fold Cross-Validation technique.
    """

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_mean, X_std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    X = (
        X - X_mean
    ) / X_std  # standardize the full (training and test) data using the statistics computed above

    # Stratified K-Fold Cross-Validation
    skf_state = jr.randint(key, shape=(), minval=0, maxval=2**15, dtype=jnp.int32)
    skf = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=skf_state.item()
    )

    dataloaders = []

    fold_keys = jr.split(key, num=2)
    for train_index, test_index in skf.split(X, y):

        train_index = train_index[
            :num_training_samples
        ]  # only use the first `num_training_samples` samples of the training indices

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_dataset = BaseDataset(X_train, y_train)
        test_dataset = BaseDataset(X_test, y_test)

        train_sampler_key, test_sampler_key = jr.split(fold_keys[-1], num=2)

        train_sampler = RandomSampler(train_sampler_key, train_dataset)
        test_sampler = RandomSampler(test_sampler_key, test_dataset)

        train_dataloader = JaxNumpyLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler
        )
        test_dataloader = JaxNumpyLoader(
            test_dataset, batch_size=len(test_dataset), sampler=test_sampler
        )

        dataloaders.append((train_dataloader, test_dataloader))

    return dataloaders


def make_pinwheel(
    key,
    num_samples,
    num_classes,
    radial_std=0.5,
    tangential_std=0.12,
    rate=0.3,
    rescale_factor=10,
):
    """
    Generate the pinwheel dataset, source:
    https://arxiv.org/pdf/1603.06277
    """
    num_per_class = int(num_samples // num_classes)
    key_feature, key_perm = jr.split(key)
    rads = jnp.linspace(0, 2 * jnp.pi, num_classes, endpoint=False)
    features = jr.normal(key_feature, shape=(num_classes * num_per_class, 2))
    features *= jnp.array([radial_std, tangential_std])
    features += jnp.array([1.0, 0.0])
    labels = jnp.repeat(jnp.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * jnp.exp(features[:, 0])
    rotations = jnp.stack(
        [jnp.cos(angles), -jnp.sin(angles), jnp.sin(angles), jnp.cos(angles)]
    )
    rotations = jnp.reshape(rotations.T, (-1, 2, 2))
    data = jnp.einsum("ti,tij->tj", features, rotations)
    return rescale_factor * data, labels


def create_pinwheel_generator(
    key,
    batch_size,
    num_training_samples,
    num_test_samples,
    num_classes,
    num_standardization_samples=1e4,
    radial_std=0.7,
    tangential_std=0.3,
    rate=0.2,
):

    key, train_data_key, test_data_key, standardization_data_key = jr.split(key, 4)
    seeds = jr.randint(key, shape=(2,), minval=0, maxval=2**15, dtype=jnp.int32)
    X, _ = make_pinwheel(
        standardization_data_key,
        num_standardization_samples,
        num_classes,
        radial_std=radial_std,
        tangential_std=tangential_std,
        rate=rate,
    )

    X_mean, X_std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)

    X_train, Y_train = make_pinwheel(
        train_data_key,
        num_training_samples,
        num_classes,
        radial_std=radial_std,
        tangential_std=tangential_std,
        rate=rate,
    )

    X_test, Y_test = make_pinwheel(
        test_data_key,
        num_test_samples,
        num_classes,
        radial_std=radial_std,
        tangential_std=tangential_std,
        rate=rate,
    )

    X_train = (
        X_train - X_mean
    ) / X_std  # standardize the full (training and test) data using the statistics computed above
    X_test = (X_test - X_mean) / X_std

    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.int32)
    X_test = np.array(X_test, dtype=np.float32)
    Y_test = np.array(Y_test, dtype=np.int32)

    train_loader_key, test_loader_key = jr.split(key)
    Dataset = BaseDataset(X_train, Y_train, transform=None)
    train_dataloader = JaxNumpyLoader(
        Dataset,
        batch_size=batch_size,
        sampler=RandomSampler(key=train_loader_key, data_source=Dataset),
    )

    Dataset = BaseDataset(X_test, Y_test, transform=None)
    test_dataloader = JaxNumpyLoader(
        Dataset,
        batch_size=num_test_samples,
        sampler=RandomSampler(key=test_loader_key, data_source=Dataset),
    )

    return train_dataloader, test_dataloader


def find_uci_stats(data_name):
    """
    This functions returns basic stats of a UCI dataset.
    """

    if data_name == "breast_cancer":
        ucirepo = fetch_ucirepo(id=17)

    elif data_name == "statlog":
        ucirepo = fetch_ucirepo(id=149)

    elif data_name == "rice":
        ucirepo = fetch_ucirepo(id=545)

    elif data_name == "waveform":
        ucirepo = fetch_ucirepo(id=107)

    elif data_name == "handwritten_digits":
        ucirepo = fetch_ucirepo(id=80)

    elif data_name == "banknote":
        ucirepo = fetch_ucirepo(id=267)

    elif data_name == "hcv":
        ucirepo = fetch_ucirepo(id=503)

    elif data_name == "mammographic":
        ucirepo = fetch_ucirepo(id=161)

    elif data_name == "dry_bean":
        ucirepo = fetch_ucirepo(id=602)

    else:
        raise ValueError(
            f"expected dataset: {data_name} is not implemented in the function."
        )

    X = ucirepo.data.features
    y = ucirepo.data.targets
    y = y.to_numpy()[:, 0]
    return {"x_dim": X.shape[1], "n_classes": len(np.unique(y))}


def create_uci_dataloader(
    key, data_name, train_size, test_size, not_split=False, max_train_size=100
):
    """
    This functions provides a generic API that can load and preprocess a UCI dataset.
    """

    int_label_list = [
        "waveform",
        "banknote",
        "hcv",
    ]

    if data_name == "breast_cancer":
        ucirepo = fetch_ucirepo(id=17)

    elif data_name == "statlog":
        ucirepo = fetch_ucirepo(id=149)

    elif data_name == "rice":
        ucirepo = fetch_ucirepo(id=545)

    elif data_name == "waveform":
        ucirepo = fetch_ucirepo(id=107)

    elif data_name == "connectionist_bench":
        ucirepo = fetch_ucirepo(id=151)

    elif data_name == "banknote":
        ucirepo = fetch_ucirepo(id=267)

    elif data_name == "hcv":
        ucirepo = fetch_ucirepo(id=503)

    elif data_name == "iris":
        ucirepo = fetch_ucirepo(id=53)

    else:
        raise ValueError(
            f"expected dataset: {data_name} is not implemented in the function."
        )

    X = ucirepo.data.features
    y = ucirepo.data.targets

    if data_name in int_label_list:
        X = X.to_numpy(dtype=float)
        y = y.to_numpy(dtype=int)[:, 0]

    else:
        X = X.to_numpy(dtype=float)
        y = y.to_numpy(dtype=str)[:, 0]

        if data_name == "statlog":
            idx_remove = np.where(y == "204")[0].item()
            X = np.delete(X, obj=idx_remove, axis=0)
            y = np.delete(y, obj=idx_remove, axis=0)

        for i, l in enumerate(np.unique(y)):
            idx = np.where(y == l)[0]
            np.put(y, idx, [i] * len(idx))

        y = y.astype(int)

    X_mean, X_std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    X = (X - X_mean) / np.maximum(X_std, 1e-12)

    permuate_key, train_sampler_key, test_sampler_key = jr.split(key, num=3)
    idx = jr.permutation(permuate_key, jnp.arange(X.shape[0]))

    X, y = X[idx], y[idx]

    if data_name == "hcv":
        y -= 1

    stats = {"x_dim": X.shape[1], "n_classes": len(np.unique(y))}

    if not_split:
        return (X, y), stats

    X_train, y_train, X_test, y_test = split_raw_data(
        X, y, train_size, test_size, max_train_size=max_train_size
    )

    Dataset_train = BaseDataset(X_train, y_train, transform=None)
    train_dataloader = JaxNumpyLoader(
        Dataset_train,
        batch_size=X_train.shape[0],
        sampler=RandomSampler(key=train_sampler_key, data_source=Dataset_train),
    )

    Dataset_test = BaseDataset(X_test, y_test, transform=None)
    test_dataloader = JaxNumpyLoader(
        Dataset_test,
        batch_size=X_test.shape[0],
        sampler=RandomSampler(key=test_sampler_key, data_source=Dataset_test),
    )

    return (train_dataloader, test_dataloader), stats


def create_breast_cancer_generator(
    key, train_size, test_size, batch_size, not_split=False, max_train_size=400
):
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    X = X.to_numpy(dtype=float)
    y = y.to_numpy(dtype=str)[:, 0]
    y = np.where(y == "M", 0, 1)

    X_mean, X_std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    X = (X - X_mean) / X_std

    # shuffle
    permuate_key, train_sampler_key, test_sampler_key = jr.split(key, num=3)

    idx = jr.permutation(permuate_key, jnp.arange(X.shape[0]))
    X = X[idx]
    y = y[idx]

    if not_split:
        return X, y

    X_train, y_train, X_test, y_test = split_raw_data(
        X, y, train_size, test_size, max_train_size=max_train_size
    )

    Dataset = BaseDataset(X_train, y_train, transform=None)
    train_dataloader = JaxNumpyLoader(
        Dataset,
        batch_size=batch_size,
        sampler=RandomSampler(key=train_sampler_key, data_source=Dataset),
    )

    Dataset = BaseDataset(X_test, y_test, transform=None)
    test_dataloader = JaxNumpyLoader(
        Dataset,
        batch_size=X_test.shape[0],
        sampler=RandomSampler(key=test_sampler_key, data_source=Dataset),
    )

    return [(train_dataloader, test_dataloader)]


def create_statlog_generator(
    key,
    train_size,
    test_size,
    batch_size,
    num_folds=10,
    not_split=False,
    max_train_size=640,
):
    statlog_vehicle_silhouettes = fetch_ucirepo(id=149)

    # data (as pandas dataframes)
    X = statlog_vehicle_silhouettes.data.features
    y = statlog_vehicle_silhouettes.data.targets
    X = X.to_numpy(dtype=float)
    y = y.to_numpy(dtype=str)[:, 0]
    idx_remove = np.where(y == "204")[0].item()
    X = np.delete(X, obj=idx_remove, axis=0)
    y = np.delete(y, obj=idx_remove, axis=0)

    for i, l in enumerate(np.unique(y)):
        idx = np.where(y == l)[0]
        np.put(y, idx, [i] * len(idx))

    y = y.astype(int)

    X_mean, X_std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    X = (X - X_mean) / X_std

    # shuffle
    permuate_key, train_sampler_key, test_sampler_key = jr.split(key, num=3)

    idx = jr.permutation(permuate_key, jnp.arange(X.shape[0]))
    X = X[idx]
    y = y[idx]

    if not_split:
        return X, y

    X_train, y_train, X_test, y_test = split_raw_data(
        X, y, train_size, test_size, max_train_size=max_train_size
    )

    Dataset = BaseDataset(X_train, y_train, transform=None)
    train_dataloader = JaxNumpyLoader(
        Dataset,
        batch_size=batch_size,
        sampler=RandomSampler(key=train_sampler_key, data_source=Dataset),
    )

    Dataset = BaseDataset(X_test, y_test, transform=None)
    test_dataloader = JaxNumpyLoader(
        Dataset,
        batch_size=X_test.shape[0],
        sampler=RandomSampler(key=test_sampler_key, data_source=Dataset),
    )

    return [(train_dataloader, test_dataloader)]


def create_rice_generator(
    key, train_size, test_size, batch_size, not_split=False, max_train_size=2560
):
    rice_cammeo_and_osmancik = fetch_ucirepo(id=545)
    X = rice_cammeo_and_osmancik.data.features
    y = rice_cammeo_and_osmancik.data.targets
    X = X.to_numpy(dtype=float)
    y = y.to_numpy(dtype=str)[:, 0]
    y = np.where(y == "Cammeo", 0, 1)

    # standardize the data
    X_mean, X_std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    X = (
        X - X_mean
    ) / X_std  # standardize the full (training and test) data using the statistics computed above

    # shuffle
    permuate_key, train_sampler_key, test_sampler_key = jr.split(key, num=3)

    idx = jr.permutation(permuate_key, jnp.arange(X.shape[0]))
    X = X[idx]
    y = y[idx]

    if not_split:
        return X, y

    X_train, y_train, X_test, y_test = split_raw_data(
        X, y, train_size, test_size, max_train_size=max_train_size
    )

    Dataset = BaseDataset(X_train, y_train, transform=None)
    train_dataloader = JaxNumpyLoader(
        Dataset,
        batch_size=batch_size,
        sampler=RandomSampler(key=train_sampler_key, data_source=Dataset),
    )

    Dataset = BaseDataset(X_test, y_test, transform=None)
    test_dataloader = JaxNumpyLoader(
        Dataset,
        batch_size=X_test.shape[0],
        sampler=RandomSampler(key=test_sampler_key, data_source=Dataset),
    )

    return [(train_dataloader, test_dataloader)]


def create_waveform_generator(
    key, train_size, test_size, batch_size, not_split=False, max_train_size=3840
):
    waveform_database_generator_version_1 = fetch_ucirepo(id=107)

    X = waveform_database_generator_version_1.data.features
    y = waveform_database_generator_version_1.data.targets
    X = X.to_numpy(dtype=float)
    y = y.to_numpy(dtype=int)[:, 0]

    # standardize the data
    X_mean, X_std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    X = (
        X - X_mean
    ) / X_std  # standardize the full (training and test) data using the statistics computed above

    # shuffle
    permuate_key, train_sampler_key, test_sampler_key = jr.split(key, num=3)
    idx = jr.permutation(permuate_key, jnp.arange(X.shape[0]))
    X = X[idx]
    y = y[idx]

    if not_split:
        return X, y

    X_train, y_train, X_test, y_test = split_raw_data(
        X, y, train_size, test_size, max_train_size=max_train_size
    )

    Dataset = BaseDataset(X_train, y_train, transform=None)
    train_dataloader = JaxNumpyLoader(
        Dataset,
        batch_size=batch_size,
        sampler=RandomSampler(key=train_sampler_key, data_source=Dataset),
    )

    Dataset = BaseDataset(X_test, y_test, transform=None)
    test_dataloader = JaxNumpyLoader(
        Dataset,
        batch_size=X_test.shape[0],
        sampler=RandomSampler(key=test_sampler_key, data_source=Dataset),
    )

    return [(train_dataloader, test_dataloader)]
