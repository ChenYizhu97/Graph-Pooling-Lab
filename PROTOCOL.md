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
pre_gnn -> pre_conv -> pool -> post_conv -> readout -> post_gnn
```

Model rules:

- `readout` is global add pooling concatenated with global max pooling.
- `pre_gnn[-1]` must equal `hidden_features`.
- `post_gnn[0]` must equal `2 * hidden_features`.
- `variant=sum` adds pre-pooling and post-pooling graph embeddings.
- `variant=plain` uses only the post-pooling graph embedding.
- `pre_conv` and `post_conv` are separate, explicit encoder roles.
- Convolution profiles describe which connectivity values a layer can consume.
  A pre-pooling convolution may use only binary topology when it cannot consume
  scalar edge values; those values are still forwarded unchanged to pooling.
- A post-pooling convolution must consume the connectivity type produced by the
  pooling method. Incompatible pooled output is rejected.

## Pool Protocol

All pooling modules must return `PoolingOutput`.

Required fields:

- `x`
- `edge_index`
- `batch`

Optional fields:

- `edge_weight`
- `perm`
- `score`
- `aux_loss`

`edge_weight` is the only scalar-connectivity channel in the GPLab pool
contract. Adapter-local names such as PyG's `edge_attr` must be converted at
the adapter boundary.

Custom pooling profiles use:

```text
<python_module>:<profile_name>
```

The referenced object must be a `PoolingProfile` containing at least one
declared `PoolingSignature` and a builder. The builder must return a pooling
module that implements `reset_parameters()`.

Pooling signatures alone define each method's valid input/output connectivity
domains; convolution capabilities do not alter those signatures.

Dense assignment pooling methods (`mincutpool`, `diffpool`, `densepool`) follow
one rule: input masks suppress padded input nodes before pooling, output nodes
are fixed cluster slots, all output cluster slots are kept, and pooled adjacency
is preserved as `edge_weight`.
