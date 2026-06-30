# Agent Reference

This file defines the machine-facing GPLab interface. The stable benchmark
rules live in [PROTOCOL.md](PROTOCOL.md).

## Static Facts

### AUTOMATION_OUTPUT_FORMATS

`["text", "json"]`

### AUTOMATION_ENTRYPOINTS

`["gplab-run-job", "gplab-normalize-job", "gplab-expand-cases", "gplab-query", "gplab-replay", "gplab-validate"]`

### SUPPORTED_DATASETS

`["MUTAG", "PROTEINS", "ENZYMES", "FRANKENSTEIN", "Mutagenicity", "AIDS", "DD", "NCI1", "COX2"]`

### BUILTIN_POOLS

`["nopool", "topkpool", "sagpool", "asapool", "sparsepool", "mincutpool", "diffpool", "densepool"]`

### CUSTOM_POOL_FORMAT

`"<python_module>:<factory_name>"`

### CORE_MODEL

The core request object is:

```text
BenchmarkRequest = BenchmarkCase + ExecutionOptions
```

`BenchmarkCase` contains only benchmark-defining fields. `ExecutionOptions`
contains `log_file`, `tag`, and `activation_checkpoint`.

### STRICT_JOB_SCHEMA

Required top-level fields:

- `case`
- `execution`

`case` fields:

- `dataset`
- `pool`
- `model`
- `training`

`case.pool` fields:

- `name`: string
- `ratio`: number in `(0, 1]`
- `nonlinearity`: non-empty string

`case.model` fields:

- `hidden_features`: integer
- `nonlinearity`: string
- `p_dropout`: number in `[0, 1)`
- `conv_layer`: string
- `pre_gnn`: integer array
- `post_gnn`: integer array
- `variant`: `"sum"` or `"plain"`

`case.training` fields:

- `runs`: integer greater than 0
- `lr`: positive finite number
- `batch_size`: integer greater than 0
- `patience`: integer greater than or equal to 0
- `epochs`: integer greater than 0
- `split`: object with `train` and `val`
- `seeds`: object with `mode`, `base`, `values`, and `allow_duplicates`

`case.training.seeds.mode` is `"auto"` or `"list"`.

`execution` fields:

- `log_file`: string or null
- `tag`: string or null
- `activation_checkpoint`: boolean

Unknown fields are rejected.

Example:

```json
{
  "case": {
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
      "conv_layer": "GCN",
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
  "execution": {
    "log_file": null,
    "tag": null,
    "activation_checkpoint": false
  }
}
```

### RECORD_SCHEMA

Records contain:

- `record_id`
- `case`
- `execution`
- `run_plan`
- `runtime`
- `result`

`run_plan` contains `case_id`, resolved `seeds`, and concrete `train` / `val` /
`test` split indices.

### SUMMARY_FIELDS

Query summaries include:

- `record_id`
- `case_id`
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
- `best_test_acc`
- `worst_test_acc`
- `val_loss_test_acc_corr`
- optional `tag`
- optional `case`
- optional `replay_command`

## Tools

### gplab-normalize-job

Validate one strict job:

```bash
gplab-normalize-job --job-file <path> --output-format json
```

Output kind: `normalized_job`.

### gplab-expand-cases

Expand combinations into complete strict jobs:

```bash
gplab-expand-cases \
  --pools <csv> \
  --datasets <csv> \
  --model-variants <csv> \
  --pool-ratio <float> \
  --output-format json
```

Relevant overrides:

- `--pool-nonlinearity`
- `--activation-checkpoint`
- `--runs`
- `--epochs`
- `--patience`
- `--lr`
- `--batch-size`
- `--split-train`
- `--split-val`
- `--seed-mode`
- `--seed-base`
- `--seed-list`
- `--allow-duplicate-seeds`
- `--log-file`
- `--tag-prefix`

Output kind: `case_manifest`.

### gplab-run-job

Execute one strict job:

```bash
gplab-run-job --job-file <path> --output-format json
```

Output kind: `train_result`.

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

Report and inspection flags:

- `--report`
- `--sort-by`
- `--show-case`
- `--show-replay`

Output kinds: `query_result`, `query_report`.

### gplab-replay

Rebuild a strict job from one record:

```bash
gplab-replay --log-file <path> --record-id <id> --output-format json
```

Use `--run` to execute the replay. Replay fixes resolved seeds as
`case.training.seeds.mode="list"` and writes resolved seeds to
`case.training.seeds.values`.

Output kind: `replay_result`.

### gplab-validate

Run caller-directed smoke validation:

```bash
gplab-validate --pools <csv> --datasets <csv> --output-format json
```

Important options:

- `--model-variant`
- `--pool-ratio`
- `--pool-nonlinearity`
- `--activation-checkpoint`
- `--runs`
- `--epochs`
- `--patience`
- `--lr`
- `--batch-size`
- `--split-train`
- `--split-val`
- `--seed-mode`
- `--seed-base`
- `--seed-list`
- `--allow-duplicate-seeds`
- `--log-file`
- `--tag-prefix`

Output kind: `validation_result`.

## Rules

- Use strict jobs for automation execution.
- Use `gplab-expand-cases` for planning combinations.
- Treat `BenchmarkCase` as the benchmark-defining object.
- Treat `ExecutionOptions` as execution-only.
- Do not derive benchmark grouping in query code; use the benchmark comparison layer.
- Dense pool output nodes are fixed cluster slots. Do not infer pruning or input-node retention.
