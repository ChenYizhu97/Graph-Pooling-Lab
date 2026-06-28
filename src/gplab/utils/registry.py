TU_DATASETS = (
    "MUTAG",
    "PROTEINS",
    "ENZYMES",
    "FRANKENSTEIN",
    "Mutagenicity",
    "AIDS",
    "DD",
    "NCI1",
    "COX2",
)

BUILTIN_POOLS = (
    "nopool",
    "topkpool",
    "sagpool",
    "asapool",
    "sparsepool",
    "mincutpool",
    "diffpool",
    "densepool",
)

DENSE_POOLS = (
    "mincutpool",
    "diffpool",
    "densepool",
)

SUPPORTED_CONVS = (
    "GCN",
    "GraphConv",
    "GIN",
)
