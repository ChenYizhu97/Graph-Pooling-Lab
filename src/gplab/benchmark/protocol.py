from __future__ import annotations

from gplab.utils.registry import BUILTIN_POOLS, DENSE_POOLS, SUPPORTED_CONVS, TU_DATASETS


TASK = "graph_classification"
DATASET_FAMILY = "TU"
DATASET_LOADER = "torch_geometric.datasets.TUDataset"
USE_NODE_ATTR = True
SUPPORTED_DATASETS = TU_DATASETS
SUPPORTED_POOLS = BUILTIN_POOLS
DENSE_POOL_METHODS = DENSE_POOLS
SUPPORTED_CONV_LAYERS = SUPPORTED_CONVS

MODEL_PATH = ("pre_gnn", "conv1", "pool", "conv2", "readout", "post_gnn")
READOUT_KIND = "add_max_concat"

POOL_REQUIRED_FIELDS = ("x", "edge_index", "batch")
POOL_OPTIONAL_FIELDS = ("edge_attr", "edge_weight", "perm", "score", "aux_loss")

TRAIN_LOSS_KIND = "classification_plus_auxiliary"
EARLY_STOPPING_METRIC = "val_classification_loss"
PRIMARY_METRIC = "best_test_acc_mean_std"

COMPARISON_INCLUDE = (
    "case.dataset",
    "case.model",
    "case.pool.ratio",
    "case.pool.nonlinearity",
    "case.training",
    "run_plan.seeds",
)

COMPARISON_EXCLUDE = (
    "case.pool.name",
    "execution",
    "runtime",
    "summary",
    "report",
)
