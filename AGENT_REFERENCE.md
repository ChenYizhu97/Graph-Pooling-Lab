# Agent Reference

This file defines the machine-facing GPLab interface. The stable benchmark
rules live in [PROTOCOL.md](PROTOCOL.md).

## Static Facts

### AUTOMATION_OUTPUT_FORMATS

`["text", "json"]`

### AUTOMATION_ENTRYPOINTS

`["gplab-run-job", "gplab-query", "gplab-replay"]`

### SUPPORTED_DATASETS

`["MUTAG", "PROTEINS", "ENZYMES", "FRANKENSTEIN", "Mutagenicity", "AIDS", "DD", "NCI1", "COX2"]`

### POOL OUTPUT

Use native `tgp.src.PoolingOutput`; GPLab only re-exports it. Pool calls use
`x`, `adj`, `batch`, and optional `edge_weight` keywords. The backbone requires
sparse output graph tensors. Selection metadata stays in `so`; auxiliary losses
are named, already-weighted scalar tensors in `loss`, summed by the classifier.
TopK/SAG are native TGP modules with a small connector ordering correction.

### POOL COMPARABILITY

Pool signatures are a non-empty set of input/output connectivity pairs. For the
actual dataset input type, every declared possible output must be supported by
the post encoder. Multiple input types are allowed; outputs are conditional on
that input, not the union of outputs from unrelated input domains.

The Python API `check_comparability(pools, ComparisonSetting(dataset, model))`
returns shared valid `input_types` plus incompatibilities grouped by input type
and pool. A verdict is true iff at least one shared input is valid. Dataset
`connectivity_type` is native/default; `connectivity_types` lists available
representations (binary: `{U}`, scalar: `{U, W}`). Every compared pool must use
the same representation; individually compatible but disjoint inputs fail.
An explicit `ComparisonSetting.input_type` prevents fallback to another type.
`load_dataset(name, connectivity_type=...)` projects scalar to binary by keeping
edges and dropping weights after existing transforms. Training entry points
still use the native default; comparability does not select runtime inputs.
`validate_pool_compatibility` checks one pool before execution. Neither API
substitutes for controlling concrete splits and training rules across runs.
Built-in scalar-input domains are not broadened merely by supporting sets.

### BUILTIN_POOLS

`["nopool", "topkpool", "sagpool", "asapool", "sparsepool", "mincutpool", "diffpool", "densepool"]`

### CUSTOM_POOL_FORMAT

`"<python_module>:<profile_name>"`

### INTERFACE_MODEL

Use Job JSON for agent-driven execution. A Job JSON request contains one
benchmark-defining `experiment` and optional execution settings.

Successful execution returns a `train_result` whose `record` is the canonical
persisted `ExperimentRecord`. Query and replay operate on JSONL logs containing
those records. An `experiment_id` identifies the requested configuration and
remains unchanged during replay. `source_record_id` identifies the source record.

### JOB_JSON_SCHEMA

A Job JSON describes exactly one experiment. Do not send arrays of
datasets, pools, ratios, or training settings; schedule those as separate
`gplab-run-job` processes.

Required top-level fields:

- `experiment`

Optional top-level fields:

- `execution`: object containing `activation_checkpoint`
- `log_file`: string or null, append destination for this job only
- `tag`: string or null
- `runs`: null or a nonempty array of `{seed, split}` objects for exact replay
- `source_record_id`: string or null, the record being replayed

Each explicit run contains an integer `seed` and a `split` object with nonempty
integer arrays `train`, `val`, and `test`. The number of runs must equal
`experiment.training.runs`. Once data is loaded, partitions must cover every
example exactly once, without overlaps or out-of-range indices. Explicit runs
bypass seed/split generation; they do not rewrite the requested seed policy.

Required `experiment` fields:

- `dataset`
- `pool`
- `training`

Optional `experiment` fields:

- `model`

Required `experiment.pool` fields:

- `name`: string
- `ratio`: number in `(0, 1]`

Optional `experiment.pool` fields:

- `nonlinearity`: non-empty string

Optional `experiment.model` fields:

- `hidden_features`: integer
- `nonlinearity`: string
- `p_dropout`: number in `[0, 1)`
- `pre_conv`: string
- `post_conv`: string
- `pre_gnn`: integer array
- `post_gnn`: integer array
- `variant`: `"sum"` or `"plain"`

Required `experiment.training` fields:

- `runs`: integer greater than 0
- `patience`: integer greater than or equal to 0
- `epochs`: integer greater than 0

Optional `experiment.training` fields:

- `lr`: positive finite number
- `batch_size`: integer greater than 0
- `split`: object with `train` and `val`
- `seeds`: object with `mode`, `base`, `values`, and `allow_duplicates`

`experiment.training.seeds.mode` is `"auto"` or `"list"`.

Optional `execution` fields:

- `activation_checkpoint`: boolean

Omitted optional fields use GPLab automation defaults. Unknown fields are
rejected.

Minimal example:

```json
{
  "experiment": {
    "dataset": "MUTAG",
    "pool": {
      "name": "nopool",
      "ratio": 0.5
    },
    "training": {
      "runs": 1,
      "epochs": 1,
      "patience": 0
    }
  }
}
```

Complete example:

```json
{
  "experiment": {
    "dataset": "PROTEINS",
    "pool": {
      "name": "sagpool",
      "ratio": 0.5,
      "nonlinearity": "tanh"
    },
    "model": {
      "hidden_features": 128,
      "nonlinearity": "relu",
      "p_dropout": 0.0,
      "pre_conv": "GCN",
      "post_conv": "GCN",
      "pre_gnn": [128],
      "post_gnn": [256, 128],
      "variant": "sum"
    },
    "training": {
      "runs": 10,
      "lr": 0.0005,
      "batch_size": 32,
      "patience": 50,
      "epochs": 500,
      "split": {
        "train": 0.8,
        "val": 0.1
      },
      "seeds": {
        "mode": "auto",
        "base": 20260320,
        "values": null,
        "allow_duplicates": false
      }
    }
  },
  "execution": {"activation_checkpoint": false},
  "log_file": null,
  "tag": null,
  "runs": null,
  "source_record_id": null
}
```

### RECORD_SCHEMA

Records are append-only JSONL entries produced by executed requests.
`ExperimentRecord` is a JSON object matching this schema:

```json
{
  "record_id": "c3433057e520",
  "experiment": {
    "...": "ExperimentConfig mapping"
  },
  "execution": {
    "...": "ExecutionOptions mapping"
  },
  "experiment_id": "f7a12815cbc5",
  "tag": null,
  "source_record_id": null,
  "environment": {
    "created_at_utc": "2026-07-02T00:00:00+00:00",
    "python_version": "3.14.7",
    "torch_version": "2.10.0+cu128",
    "torch_geometric_version": "2.8.0.post1",
    "device": "cpu",
    "cuda_available": false,
    "cudnn_deterministic": true,
    "cudnn_benchmark": false
  },
  "result": {
    "mean": 0.5,
    "std": 0.0,
    "trainable_parameters": {
      "total": 1000,
      "pooling_module": 17
    },
    "runs": [
      {
        "seed": 457750178,
        "split": {"train": [0, 1], "val": [2], "test": [3]},
        "best_epoch": 1,
        "best_val_loss": 1.0,
        "best_val_auxiliary_loss": 0.0,
        "test_acc": 0.5,
        "training_wall_time_seconds": 0.25,
        "epochs_trained": 1,
        "peak_training_cuda_allocated_bytes": null,
        "structural_stats": {
          "total_input_nodes": 10,
          "total_output_nodes": 6,
          "total_input_edges": 20,
          "total_output_edges": 8,
          "total_input_nonzero_edges": 20,
          "total_output_nonzero_edges": 6,
          "num_graphs": 2,
          "mean_node_retention": 0.6
        }
      }
    ]
  }
}
```

`result.runs` stores each seed and its concrete train/validation/test indices
alongside the measurements. `result.mean` and `result.std` are computed from
`result.runs[*].test_acc`, the accuracy from the single final test evaluation of
each run's best validation checkpoint.

`result.trainable_parameters` is counted after all runs, when lazy parameters
have been initialized by training, and is invariant across runs. Run-level wall
time and epoch count cover the training loop, including validation passes but
excluding the one final test pass. `peak_training_cuda_allocated_bytes` is
`null` on CPU; on CUDA it records the peak allocated memory during the epoch
loop, including validation and checkpoint saving. The peak is read after
training finishes and CUDA is synchronized, before checkpoint restoration,
final test evaluation, or structural-statistics instrumentation.
`structural_stats` contains only aggregates from that final test pass. Node
retention is averaged per graph rather than computed as a ratio of the two node
totals, and no per-graph observations are written to the record.

`total_input_edges` and `total_output_edges` count stored connectivity entries,
including zero weights. The corresponding `total_input_nonzero_edges` and
`total_output_nonzero_edges` count strictly nonzero weights; without edge
weights, every stored entry counts. Directions, self-loops, and duplicates are
counted as stored, and no small-weight threshold is applied. For example,
`total_output_nonzero_edges / total_output_edges` gives the aggregate nonzero
fraction when the denominator is positive; leave it undefined for zero edges.

One record log line is one `ExperimentRecord`. `gplab-query` and `gplab-replay`
both consume this JSONL format; malformed records return structured config
errors instead of being treated as partial records.

Replay rebuilds an `ExperimentJob` from `experiment`, `execution`, and the
seed/split pairs in `result.runs`. The requested seed policy remains unchanged.
The replay job's `runs` field supplies the exact recorded repetitions, while
`source_record_id` preserves provenance. The source log destination is never reused.

### SUMMARY_FIELDS

`max_test_acc` and `min_test_acc` are the maximum and minimum of the per-run
`test_acc` values across runs.

Query summaries include:

- `record_id`
- `experiment_id`
- `benchmark_key`
- `dataset`
- `pool`
- `pool_ratio`
- `pool_nonlinearity`
- `activation_checkpoint`
- `model_variant`
- `runs`
- `mean`
- `std`
- `avg_best_epoch`
- `avg_val_loss`
- `avg_val_auxiliary_loss`
- `max_test_acc`
- `min_test_acc`
- `val_loss_test_acc_corr`
- optional `tag`
- optional `experiment`
- optional `replay_command`

## Tools

Recommended agent workflow:

```text
agent writes job.json -> gplab-run-job -> gplab-query
                                  \
                                   -> gplab-replay
```

`gplab-run-job` is also the validation boundary. If the job is invalid, it exits
non-zero and returns `ok=false` with `error.type="config_error"` and a
field-specific `error.message`. When `--output-format json` is used, stdout is
reserved for the single JSON response; progress and third-party output go to
stderr.

### JSON_OUTPUT_CONTRACT

For every GPLab CLI invocation using `--output-format json`, stdout contains
exactly one JSON object on success or on handled errors. Progress, diagnostics,
dependency output, and shell wrappers belong to stderr. Agents should parse
stdout as the machine response and use the process exit code only to distinguish
success from failure.

### ERROR_RESPONSE

Handled failures use this envelope:

```json
{
  "ok": false,
  "kind": "job_error",
  "error": {
    "type": "config_error",
    "message": "Human-readable error.",
    "field": "experiment.pool.ratio",
    "expected": "finite number",
    "missing": ["ratio"],
    "unknown": ["extra"],
    "details": {
      "source": "job_json"
    }
  }
}
```

`kind` is command-specific, such as `job_error`, `query_error`, or
`replay_error`. `error.type` is one of `config_error`, `file_not_found`, or
`runtime_error`. `field`, `expected`, `missing`, `unknown`, and `details` are
included only when available.

### gplab-run-job

Execute one Job JSON request:

```bash
gplab-run-job --job-file <path> --output-format json
```

Other input forms:

```bash
gplab-run-job --job-json '<json>' --output-format json
cat job.json | gplab-run-job --job-stdin --output-format json
```

Provide exactly one of `--job-file`, `--job-json`, or `--job-stdin`.

Output kind: `train_result` on success. Success responses contain:

- `record`: the canonical `ExperimentRecord`
- `summary`: a derived result summary
- `context`: command context with `source="job_json"` and `experiment_id`; `job_file`
  is included only for file input

Invalid jobs return kind `job_error`.

Invalid job response shape:

```json
{
  "ok": false,
  "kind": "job_error",
  "error": {
    "type": "config_error",
    "message": "Missing required experiment.pool field(s): ratio.",
    "field": "experiment.pool",
    "expected": "required fields: name, ratio",
    "missing": ["ratio"],
    "details": {
      "job_file": "job.json",
      "source": "job_json"
    }
  }
}
```

### gplab-query

Read JSONL records:

```bash
gplab-query --log-file <path> --output-format json
```

Filters:

- `--pool`
- `--dataset`
- `--model-variant`
- `--tag`

Report, sort, and inspection flags:

- `--report`
- `--sort-by`
- `--show-experiment`
- `--show-replay`

Output kinds: `query_result`, `query_report`.

`query_result` contains `context` and `summaries`. `summaries` are derived
record views, not canonical `ExperimentRecord` objects. `context` reports
`source="record_log"`, `log_file`, active `filters`, `sort_by`,
`total_records`, and `matched_records`.

`query_report` contains the same `context` plus grouped benchmark comparisons.
Each group contains `benchmark_key`, a `comparison` block, and ranked
`summaries`.

### gplab-replay

Rebuild a Job JSON request from one record:

```bash
gplab-replay --log-file <path> --record-id <id> --output-format json
```

Use `--run` to execute the replay. The exported Job preserves the original
configuration and includes concrete seeds and splits in its top-level `runs`.
It can also be submitted directly to `gplab-run-job`.

Output kind: `replay_result`. The top-level `job` is the replayable Job JSON,
and `context` contains `source="record_replay"`, `source_record_id`, and the
unchanged `experiment_id`. If `--run` is used, `rerun.payload` is a standard
`train_result`.

## Rules

- Use Job JSON for automation execution.
- Execute one Job JSON request per `gplab-run-job` process.
- Let the caller schedule multiple experiments as multiple processes.
- Do not let multiple processes append to the same `log_file`; use
  separate JSONL files or serialize writes externally.
- Treat `record` in `train_result` as the canonical persisted object.
- Treat `summary` and `context` as derived response metadata.
- Treat `query_result.summaries` and `query_report.groups[].summaries` as
  derived views, not persisted records.
- Treat `gplab-train` as a human convenience entrypoint, not the agent protocol.
- Treat `ExperimentConfig` as the benchmark-defining object.
- Treat `ExecutionOptions` as execution-only.
- Do not derive benchmark grouping in query code; use the benchmark comparison layer.
- Dense pool output nodes are fixed cluster slots. Do not infer pruning or input-node retention.
