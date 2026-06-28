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
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if not (0.0 < train_ratio < 1.0) or not (0.0 < val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be in (0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be smaller than 1")

    rng = np.random.default_rng(seed)
    rnd_idx = rng.permutation(dataset_size).tolist()

    train_end = int(train_ratio * dataset_size)
    val_end = int((train_ratio + val_ratio) * dataset_size)

    return {
        "train": rnd_idx[:train_end],
        "val": rnd_idx[train_end:val_end],
        "test": rnd_idx[val_end:],
    }


def split_dataset(
    dataset: Dataset,
    split_indices: dict,
):
    train_dataset = dataset[split_indices["train"]]
    val_dataset = dataset[split_indices["val"]]
    test_dataset = dataset[split_indices["test"]]

    return train_dataset, val_dataset, test_dataset
