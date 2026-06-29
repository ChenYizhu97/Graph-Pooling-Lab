import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset

from gplab.utils.validation import validate_dataset_value


def load_dataset(dataset: str) -> Dataset:
    validate_dataset_value(dataset)
    return TUDataset(root="/tmp/TUDataset", name=dataset, use_node_attr=True)


def build_split_indices(
    dataset_size: int,
    seed: int,
    split_train: float = 0.8,
    split_val: float = 0.1,
) -> dict:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if not (0.0 < split_train < 1.0) or not (0.0 < split_val < 1.0):
        raise ValueError("split_train and split_val must be in (0, 1)")
    if split_train + split_val >= 1.0:
        raise ValueError("split_train + split_val must be smaller than 1")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(dataset_size).tolist()

    train_end = int(split_train * dataset_size)
    val_end = int((split_train + split_val) * dataset_size)

    return {
        "train": shuffled_indices[:train_end],
        "val": shuffled_indices[train_end:val_end],
        "test": shuffled_indices[val_end:],
    }


def split_dataset(
    dataset: Dataset,
    split_indices: dict,
):
    train_dataset = dataset[split_indices["train"]]
    val_dataset = dataset[split_indices["val"]]
    test_dataset = dataset[split_indices["test"]]

    return train_dataset, val_dataset, test_dataset
