# GPLab Benchmark Protocol

This file defines the stable benchmark core for GPLab. CLI arguments, JSON jobs,
records, summaries, and reports are adapters around this protocol.

## Core Unit

The core unit is an `ExperimentConfig`: one graph-pooling benchmark experiment under a
shared graph-classification protocol.

```text
ExperimentConfig =
  dataset
  model
  pool
  training
```

`TrainingConfig` includes `activation_checkpoint`, which trades compute for memory.
`ExperimentJob` combines the configuration with `log_file`, `tag`, optional fixed runs,
and replay provenance. Those job fields do not belong to the training loop.

## Execution Flow

1. Parse CLI/TOML or JSON into an `ExperimentJob`.
2. Configure threads and random state, select the device, and capture `environment`.
3. `prepare_experiment` loads data, checks pool compatibility, and resolves runs.
4. Each `RunSpec(seed, split)` drives one reset/train/validate/final-test cycle.
5. Aggregate measurements; the job boundary builds the final record and optionally
   appends it to JSONL before returning the CLI response.

`PreparedExperiment` contains configuration, loaded data, dataset statistics,
and validated runs. Model construction reads training settings from that configuration
and receives the device explicitly. Environment metadata records only Python, Torch,
PyG, and TGP versions plus the actual device; it never controls training.

Every run stores its actual seed and split alongside its measurements in
`result.runs`. Replay copies those into the job's `runs` field, bypassing generation
without rewriting the original configuration.
`source_record_id` identifies the replay's source; `record_id` hashes the completed
record. Split validation rejects empty, repeated, overlapping, missing, and
out-of-range indices before training.

`comparison_group_key` is computed for query summaries and reports, not stored
in experiment records. It groups records with the same dataset, model, pooling
ratio/activation, training budget, and actual seeds and splits, excluding pool name.
Activation checkpointing is excluded from accuracy grouping; resource comparisons
must account for it separately. This grouping is independent of structural
comparability checks and does not establish identical data contents or environments.

## Data Protocol

- Task: graph classification.
- Dataset family: TU datasets through `torch_geometric.datasets.TUDataset`.
- Loader option: `use_node_attr=True`.
- Dataset names are restricted to the project whitelist.
- Each run builds a seeded train/validation/test split.
- The test fraction is derived as `1 - split.train - split.val`.

## Evaluation Protocol

- Training evaluates the validation split after each epoch.
- Validation classification loss alone selects the best checkpoint; validation
  auxiliary loss is recorded but is not part of the selection criterion.
- After training or early stopping, the selected checkpoint is restored and the
  test split is evaluated exactly once. Its accuracy is recorded as `test_acc`.
- Training wall time includes the epoch loop and its validation passes, but not
  the final test evaluation.
- `peak_training_cuda_allocated_bytes` measures peak training CUDA allocated
  memory. The peak is reset immediately before the epoch loop and read
  immediately after that loop finishes and CUDA is synchronized, before
  checkpoint restoration or final testing. It includes training, per-epoch
  validation, and checkpoint saving, but excludes final test evaluation and
  structural-statistics instrumentation. It is `null` on CPU.
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

All benchmark experiments use one shared backbone shape:

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

All pooling modules accept keyword arguments `x`, `adj`, `batch`, and optional
`edge_weight`, and return the native `tgp.src.PoolingOutput`. GPLab re-exports
that class; it does not define a second output container.

Required fields:

- `x`
- `edge_index`
- `batch`

Optional fields:

- `edge_weight`
- `so`: native TGP selection/assignment information, when available
- `loss`: a dictionary of named, already-weighted scalar tensors

The classifier sums `loss.values()` with gradients intact, or returns no
auxiliary loss when the dictionary is absent/empty. Existing coefficients remain
`0.5 * mincut + orthogonality` and `0.1 * link + 0.1 * entropy`. The shared
backbone requires `x: [N,F]`, `edge_index: [2,E]`, and `batch: [N]`; current
dense-method adapters still emit sparse graph batches with fixed cluster slots.
TGP's optional `so`/derived mask are not fabricated for backends that do not
expose assignments.

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

Pooling signatures form a non-empty set of `(input_type, output_type)` pairs.
A pool may accept several input types and declare several possible outputs for
one input. Signatures remain conditional on input: `{U -> U, W -> W}` does not
imply `U -> W`. Convolution capabilities do not alter those declarations.

For dataset type `t` and pool signature relation `S`, compatibility requires:

```text
t in {input_type for (input_type, output_type) in S}
{output_type for (input_type, output_type) in S if input_type == t}
    <= post_conv.connectivity_types
```

Every possible output for the actual input must be supported, not merely one
member of that set. A topology-only pre-convolution may still pass scalar
connectivity unchanged to pooling, as specified in the model protocol.

`benchmark.compatibility` checks a single pool and lists compatible built-ins.
`benchmark.comparability.check_comparability(pools, setting)` checks at least two
distinct profiles under one shared `ComparisonSetting(dataset, model, input_type=None)`.
Dataset profiles distinguish their native/default `connectivity_type` from
available `connectivity_types`: binary provides `{U}`, scalar provides `{U, W}`.
The scalar-to-binary projection preserves supplied edges and discards weights;
it does not infer topology by thresholding values. Adding ones to binary edges
does not establish semantic scalar connectivity.

For dataset representations `D` and compared pools `P`, the valid inputs are:

```text
candidates = D intersect intersection(pool.input_types for pool in P)
valid_inputs = {t in candidates where every pool's outputs(t)
                are supported by the post-convolution}
comparable = bool(valid_inputs)
```

`ComparabilityResult.input_types` contains these shared valid inputs;
`incompatibilities` groups rejected alternatives by input type, then pool.
Rejected alternatives do not invalidate another shared valid input. An explicit
`setting.input_type` restricts the check to that representation, without fallback.
Different pools need not produce the same graph type if the shared
post-convolution supports all of them.

All compared runs must select the **same** valid input representation.
`load_dataset(name, connectivity_type=...)` materializes an explicit choice;
omission uses the native default. The existing training entry points still use
that default; a structural verdict does not change their inputs automatically.
This is a structural verdict based on declared domains; experiments must also
share dataset instances, concrete splits, and training/model-selection rules.
The existing record grouping key is a protocol grouping aid, not proof that
all these execution conditions hold.

Built-in domains remain conservative, following `audits/COMPARABILITY_ALIGNMENT.md`:
`nopool` declares `{U -> U, W -> W}`; TopK, SAG, and sparsepool declare `{U -> U}`;
ASAP, DiffPool, MinCut, and densepool declare `{U -> W}`. Accepting an
`edge_weight` argument alone is not evidence of a method-faithful W-input path.

TopK and SAG use native TGP 1.0.2 modules with a connector edge-order fix. TopK scores are the
configured activation of the normalized learned feature projection, including
when the feature width is one. SAG scores are `tanh(GCNConv(X, A))`, with no
additional learned selection projection; its scorer and activation are fixed.
Both retain induced edges, including existing self-loops, without degree or
edge-weight normalization. These backend choices do not broaden their declared
input/output domains. Other pooling implementations are unchanged.

Dense assignment pooling methods (`mincutpool`, `diffpool`, `densepool`) follow
one rule: input masks suppress padded input nodes before pooling, output nodes
are fixed cluster slots, all output cluster slots are kept, and pooled adjacency
is preserved as `edge_weight`.
