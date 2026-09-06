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

## Evaluation Protocol

- Training evaluates the validation split after each epoch.
- Validation classification loss alone selects the best checkpoint; validation
  auxiliary loss is recorded but is not part of the selection criterion.
- After training or early stopping, the selected checkpoint is restored and the
  test split is evaluated exactly once.
- Training wall time includes the epoch loop and its validation passes, but not
  the final test evaluation.
- When CUDA is used, peak allocated memory is reset immediately before the epoch
  loop and read after the final test evaluation.
- Trainable parameter counts are recorded for the whole model and separately
  for the pooling module, after all runs finish so normal training has
  initialized any lazy parameters.
- The final test pass records aggregate input/output node and edge totals and
  graph count. Mean node retention is the unweighted mean of
  `output_nodes / input_nodes` over test graphs; no per-graph values are stored.
  For fixed-cluster dense methods this is an effective output-size ratio and may
  exceed one, not a claim that input nodes were literally selected.
- `total_input_edges` and `total_output_edges` count stored `edge_index`
  entries, including zero-weight entries in dense pooled graphs.
  `total_input_nonzero_edges` and `total_output_nonzero_edges` count entries
  whose `edge_weight != 0`; without edge weights, all stored entries count.
  Both counts retain directions, self-loops, and duplicates as stored, with no
  threshold for small weights. The nonzero fraction can be computed as
  `total_output_nonzero_edges / total_output_edges` when the denominator is
  positive; an empty edge set has an undefined fraction.

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

Benchmark measurements are collected by the experiment runner and are not part
of `PoolingOutput`. Pooling modules remain responsible only for producing the
pooled graph.

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
