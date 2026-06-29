# GPLab Benchmark Protocol

This file defines the stable benchmark core for GPLab. CLI arguments, JSON jobs,
records, summaries, and reports are adapters around this protocol.

## Core Unit

The core unit is a `BenchmarkCase`: one graph-pooling benchmark case under a
shared graph-classification protocol.

```text
BenchmarkCase =
  dataset
  model
  pool
  training
```

Execution-only choices such as `log_file`, `tag`, and `activation_checkpoint`
belong to `ExecutionOptions`, not to the benchmark case.

## Data Protocol

- Task: graph classification.
- Dataset family: TU datasets through `torch_geometric.datasets.TUDataset`.
- Loader option: `use_node_attr=True`.
- Dataset names are restricted to the project whitelist.
- Each run builds a seeded train/validation/test split.
- `split.test` is derived as `1 - split.train - split.val`.

## Model Protocol

All benchmark cases use one shared backbone shape:

```text
pre_gnn -> conv1 -> pool -> conv2 -> readout -> post_gnn
```

Model rules:

- `readout` is global add pooling concatenated with global max pooling.
- `pre_gnn[-1]` must equal `hidden_features`.
- `post_gnn[0]` must equal `2 * hidden_features`.
- `variant=sum` adds pre-pooling and post-pooling graph embeddings.
- `variant=plain` uses only the post-pooling graph embedding.

## Pool Protocol

All pooling modules must return `PoolOutput`.

Required fields:

- `x`
- `edge_index`
- `batch`

Optional fields:

- `edge_attr`
- `edge_weight`
- `perm`
- `score`
- `aux_loss`

Custom pooling factories use:

```text
<python_module>:<factory_name>
```

The returned module must implement `reset_parameters()`.

Dense assignment pooling methods (`mincutpool`, `diffpool`, `densepool`) follow
one rule: input masks suppress padded input nodes before pooling, output nodes
are fixed cluster slots, all output cluster slots are kept, and pooled adjacency
is preserved as `edge_weight`.

## Training And Evaluation Protocol

- Every seed is one independent run.
- The model is reset before each run.
- Training loss is `classification_loss + auxiliary_loss`.
- Validation records classification loss and auxiliary loss separately.
- Early stopping uses validation classification loss only.
- Test evaluation runs only when validation classification loss improves.
- Each run records `best_epoch`, `best_val_loss`,
  `best_val_auxiliary_loss`, and `best_test_acc`.
- The primary benchmark metric is mean/std of per-run `best_test_acc`.

## Comparison Protocol

Benchmark grouping includes:

- `case.dataset`
- `case.model`
- `case.pool.ratio`
- `case.pool.nonlinearity`
- `case.training`
- resolved `run_plan.seeds`

Benchmark grouping excludes:

- `case.pool.name`
- `execution`
- `runtime`
- summaries and reports

This means pool methods are compared inside the same benchmark group, while
execution options such as activation checkpointing do not define a different
benchmark case.
