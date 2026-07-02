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

Executable requests enter through one of three adapters:

```text
CLI / TOML options -> BenchmarkRequest
Job JSON          -> BenchmarkRequest
ExperimentRecord  -> BenchmarkRequest
```

Use Job JSON for agent-driven execution. Use records only as persisted
experiment output or as replay input. `BenchmarkRequest.to_mapping()` is the
Job JSON projection used by replay output, and `BenchmarkRequest.case_id` is
the stable case identifier.

### JOB_JSON_SCHEMA

A Job JSON describes exactly one experiment case. Do not send arrays of
datasets, pools, ratios, or training settings; schedule those as separate
`gplab-run-job` processes.

Required top-level fields:

- `case`

Optional top-level fields:

- `execution`

Required `case` fields:

- `dataset`
- `pool`
- `training`

Optional `case` fields:

- `model`

Required `case.pool` fields:

- `name`: string
- `ratio`: number in `(0, 1]`

Optional `case.pool` fields:

- `nonlinearity`: non-empty string

Optional `case.model` fields:

- `hidden_features`: integer
- `nonlinearity`: string
- `p_dropout`: number in `[0, 1)`
- `conv_layer`: string
- `pre_gnn`: integer array
- `post_gnn`: integer array
- `variant`: `"sum"` or `"plain"`

Required `case.training` fields:

- `runs`: integer greater than 0
- `patience`: integer greater than or equal to 0
- `epochs`: integer greater than 0

Optional `case.training` fields:

- `lr`: positive finite number
- `batch_size`: integer greater than 0
- `split`: object with `train` and `val`
- `seeds`: object with `mode`, `base`, `values`, and `allow_duplicates`

`case.training.seeds.mode` is `"auto"` or `"list"`.

Optional `execution` fields:

- `log_file`: string or null
- `tag`: string or null
- `activation_checkpoint`: boolean

Omitted optional fields use GPLab automation defaults. Unknown fields are
rejected.

Minimal example:

```json
{
  "case": {
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

Records are append-only JSONL entries produced by executed requests. They
contain:

- `record_id`
- `case`
- `execution`
- `run_plan`
- `runtime`
- `result`

`run_plan` contains `case_id`, resolved `seeds`, and concrete `train` / `val` /
`test` split indices.

Replay rebuilds a request from `case`, `execution`, and `run_plan.seeds`; the
replay job uses `case.training.seeds.mode="list"`. A replay result reports both
the source record case id and the replay job case id.

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
- `context`: command context with `source="job_json"` and `case_id`; `job_file`
  is included only for file input

Invalid jobs return kind `job_error`.

Invalid job response shape:

```json
{
  "ok": false,
  "kind": "job_error",
  "error": {
    "type": "config_error",
    "message": "Missing required case.pool field(s): ratio.",
    "field": "case.pool",
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
- `--show-case`
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

Use `--run` to execute the replay. Replay fixes resolved seeds as
`case.training.seeds.mode="list"` and writes resolved seeds to
`case.training.seeds.values`.

Output kind: `replay_result`. The top-level `job` is the replayable Job JSON,
and `context` contains `source="record_replay"`, `source_case_id`, and the
replay `case_id`. If `--run` is used, `rerun.payload` is a standard
`train_result`.

## Rules

- Use Job JSON for automation execution.
- Execute one Job JSON request per `gplab-run-job` process.
- Let the caller schedule multiple experiment cases as multiple processes.
- Do not let multiple processes append to the same `execution.log_file`; use
  separate JSONL files or serialize writes externally.
- Treat `record` in `train_result` as the canonical persisted object.
- Treat `summary` and `context` as derived response metadata.
- Treat `query_result.summaries` and `query_report.groups[].summaries` as
  derived views, not persisted records.
- Treat `gplab-train` as a human convenience entrypoint, not the agent protocol.
- Treat `BenchmarkCase` as the benchmark-defining object.
- Treat `ExecutionOptions` as execution-only.
- Do not derive benchmark grouping in query code; use the benchmark comparison layer.
- Dense pool output nodes are fixed cluster slots. Do not infer pruning or input-node retention.
