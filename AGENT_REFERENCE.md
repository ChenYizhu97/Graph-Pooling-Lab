# Agent Reference

## Purpose
This file defines stable facts, approved tool surfaces, and execution rules for automation clients operating GPLab. Read it before composing machine-facing train, query, replay, normalization, expansion, or validation workflows.

## Static Facts

### AUTOMATION_OUTPUT_FORMATS
- Type: string[]
- Value: `["text", "json"]`
- Meaning: Supported presentation modes on automation-capable entrypoints.
- Use: Select `json` for machine consumption and `text` only when readable console output is explicitly required.
- Do not infer: Do not assume any format other than `text` and `json` is supported.

### AUTOMATION_ENTRYPOINTS
- Type: string[]
- Value: `["gplab-run-job", "gplab-normalize-job", "gplab-expand-cases", "gplab-query", "gplab-replay", "gplab-validate"]`
- Meaning: Approved machine-facing console commands.
- Use: Build automation workflows from these entrypoints.
- Do not infer: Do not assume `gplab-train` is the preferred automation surface. It remains primarily a human-oriented composition entrypoint.

### SUPPORTED_DATASETS
- Type: string[]
- Value: `["MUTAG", "PROTEINS", "ENZYMES", "FRANKENSTEIN", "Mutagenicity", "AIDS", "DD", "NCI1", "COX2"]`
- Meaning: Built-in dataset whitelist.
- Use: Validate dataset names in jobs, manifests, and smoke plans.
- Do not infer: Do not assume datasets outside this list are supported.

### BUILTIN_POOLS
- Type: string[]
- Value: `["nopool", "topkpool", "sagpool", "asapool", "sparsepool", "mincutpool", "diffpool", "densepool"]`
- Meaning: Built-in pooling method whitelist.
- Use: Validate pooling names and generate manifests.
- Do not infer: Do not assume other built-in pool names exist.

### CUSTOM_POOL_FORMAT
- Type: string
- Value: `"<python_module>:<factory_name>"`
- Meaning: Required string format for custom pooling factories.
- Use: Pass custom pooling through job files or tool flags that accept pool names. The factory must return `torch.nn.Module`, the module must return `PoolOutput`, and it must implement `reset_parameters()`.
- Do not infer: Do not assume arbitrary plugin naming schemes are accepted.

### BENCHMARK_GROUPING_BOUNDARY
- Type: object
- Value: `{"include": ["spec.dataset", "spec.model", "spec.pool.ratio", "spec.pool.nonlinearity", "spec.train except activation_checkpoint"], "exclude": ["spec.pool.name", "spec.pool.source", "spec.train.activation_checkpoint"]}`
- Meaning: Fair-comparison grouping boundary used by query summaries and reports.
- Use: Group benchmark-equivalent records and compare pooling methods inside the same boundary.
- Do not infer: Do not treat pool method choice or activation checkpointing as part of the benchmark key. Pool ratio and pooling nonlinearity are part of the key.

### DENSE_POOL_PROTOCOL
- Type: object
- Value: `{"methods": ["mincutpool", "diffpool", "densepool"], "input_mask_applies_before_pooling": true, "output_nodes_are_clusters": true, "keep_all_output_cluster_slots": true, "preserve_pooled_adjacency_as_edge_weight": true}`
- Meaning: Dense pooling semantics in GPLab.
- Use: Interpret dense pooled outputs as fixed cluster slots, not retained input nodes.
- Do not infer: Do not infer per-cluster pruning or output-slot invalidation from weak assignments.

### EXPAND_CASES_DEFAULTS
- Type: object
- Value: `{"dataset": "PROTEINS", "pool": {"name": "nopool", "ratio": 0.5, "nonlinearity": "tanh"}, "model": {"hidden_features": 128, "nonlinearity": "relu", "p_dropout": 0.0, "conv_layer": "GCN", "pre_gnn": [128], "post_gnn": [256, 128], "variant": "sum"}, "train": {"runs": 10, "lr": 0.0005, "batch_size": 32, "patience": 50, "epochs": 500, "train_ratio": 0.8, "val_ratio": 0.1, "seed_mode": "auto", "seed_base": 20260320, "seed_list": null, "allow_duplicate_seeds": false, "activation_checkpoint": false}, "log_file": null, "tag": null}`
- Meaning: Internal defaults used when `gplab-expand-cases` materializes complete jobs.
- Use: Expect emitted manifest jobs to be complete even when callers omit optional overrides.
- Do not infer: Do not assume `gplab-normalize-job` accepts partial jobs or fills missing fields.

### TRAIN_JOB_SCHEMA
- Type: object
- Meaning: Required schema for `--job-file` consumed by `gplab-normalize-job` and `gplab-run-job`.
- Use: Provide one complete train job JSON object.
- Do not infer: Do not omit required sections or rely on missing-field defaults.
- Required top-level fields:
  - `dataset`
  - `pool`
  - `model`
  - `train`
  - `log_file`
  - `tag`
- `pool` fields:
  - `name`: string
  - `ratio`: number in `(0, 1]`
  - `nonlinearity`: non-empty string
- `model` fields:
  - `hidden_features`: integer
  - `nonlinearity`: string
  - `p_dropout`: number
  - `conv_layer`: string
  - `pre_gnn`: integer array
  - `post_gnn`: integer array
  - `variant`: `"sum"` or `"plain"`
- `train` fields:
  - `runs`: integer greater than 0
  - `lr`: number
  - `batch_size`: integer
  - `patience`: integer
  - `epochs`: integer
  - `train_ratio`: number
  - `val_ratio`: number
  - `seed_mode`: `"auto"` or `"list"`
  - `seed_base`: integer
  - `seed_list`: `null` or non-empty integer array
  - `allow_duplicate_seeds`: boolean
  - `activation_checkpoint`: boolean
- Cross-field rules:
  - `train_ratio > 0`
  - `val_ratio > 0`
  - `train_ratio + val_ratio < 1`
  - if `seed_list` is not `null`, `seed_mode` must be `"list"`
  - `pre_gnn[-1]` must equal `hidden_features`
  - `post_gnn[0]` must equal `2 * hidden_features`
- Validation behavior:
  - unknown fields are rejected
  - partial jobs are rejected
  - unsupported dataset, pool, and model variant values are rejected
- Example:

```json
{
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
  "train": {
    "runs": 10,
    "lr": 0.0005,
    "batch_size": 32,
    "patience": 50,
    "epochs": 500,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "seed_mode": "auto",
    "seed_base": 20260320,
    "seed_list": null,
    "allow_duplicate_seeds": false,
    "activation_checkpoint": false
  },
  "log_file": null,
  "tag": null
}
```

### TRAIN_RESULT_SCHEMA
- Type: object
- Value: `{"required_success_fields": ["ok", "kind", "record", "summary", "request"], "required_error_fields": ["ok", "kind", "error.type", "error.message"]}`
- Meaning: Public payload contract for train execution.
- Use: Validate `gplab-run-job` outputs and `gplab-replay` rerun payloads.
- Do not infer: Do not assume undocumented top-level fields are stable.

### QUERY_RESULT_SCHEMA
- Type: object
- Value: `{"required_fields": ["ok", "kind", "records"], "record_summary_fields": ["record_id", "benchmark_key", "dataset", "pool", "pool_ratio", "pool_nonlinearity", "activation_checkpoint", "model_type", "runs", "mean", "std", "avg_best_epoch", "avg_val_loss", "avg_val_auxiliary_loss", "best_test_acc", "worst_test_acc", "val_loss_test_acc_corr"], "optional_record_fields": ["tag", "spec", "replay_command"]}`
- Meaning: Public payload contract for flat query responses.
- Use: Parse record summaries without scraping text output.
- Do not infer: Do not assume extra analytical labels exist.

### QUERY_REPORT_SCHEMA
- Type: object
- Value: `{"required_fields": ["ok", "kind", "groups"], "group_fields": ["benchmark_key", "dataset", "model_type", "pool_ratio", "pool_nonlinearity", "records"]}`
- Meaning: Public payload contract for grouped query reports.
- Use: Consume comparable benchmark groups programmatically.
- Do not infer: Do not assume project-owned recommendation labels exist.

### REPLAY_RESULT_SCHEMA
- Type: object
- Value: `{"required_fields": ["ok", "kind", "record", "job", "execution", "paths", "compatibility"], "execution_fields": ["mode", "case_id"], "optional_rerun_fields": ["rerun.requested", "rerun.ok", "rerun.payload", "rerun.record_id", "rerun.summary", "rerun.appended_to_log"]}`
- Meaning: Public payload contract for replay responses.
- Use: Consume replay metadata and rerun handoff data without parsing prose.
- Do not infer: Do not assume replay judges metric equivalence automatically.

### VALIDATION_RESULT_SCHEMA
- Type: object
- Value: `{"required_fields": ["ok", "kind", "mode", "plan", "cases", "summary"], "case_fields": ["case_id", "pool", "pool_nonlinearity", "dataset", "model_type", "activation_checkpoint", "status", "seconds", "execution"], "optional_case_fields": ["record_id", "error_type", "message"]}`
- Meaning: Public payload contract for smoke validation responses.
- Use: Consume validation plans and results as structured data.
- Do not infer: Do not assume validation encodes project-owned policy about what must be run.

## Tools

### Read-only Tools

#### gplab-normalize-job
- Purpose: Validate a job file and emit the canonical complete request object.
- Command: `gplab-normalize-job --job-file <path> --output-format json`
- Inputs:
  - `--job-file`: path to one job JSON object
  - `--output-format`: `json` or `text`
- Output:
  - stdout success: `normalized_job` payload with `case_id` and canonical `job`
  - stdout failure: `normalize_job_error` payload when `--output-format json`
  - exit code 0: normalization succeeded
  - exit code non-zero: schema validation failed or file loading failed
- Side effects:
  - none
- Use when:
  - you need a stable executable request object before training
  - you need strict schema validation before execution
  - you need to validate job syntax without running training
- Do not use when:
  - you already have a normalized job object in memory
  - you intend to execute the job immediately and do not need a separate normalization step

#### gplab-expand-cases
- Purpose: Expand caller-provided pool/dataset/model combinations into a case manifest without execution.
- Command: `gplab-expand-cases --pools <csv> --datasets <csv> --model-types <csv> --pool-ratio <float> --output-format json`
- Inputs:
  - `--pools`: comma-separated pool list
  - `--datasets`: comma-separated dataset list
  - `--model-types`: comma-separated model variant list
  - `--pool-ratio`: pooling ratio for every case
  - `--pool-nonlinearity`: pooling score nonlinearity for every case
  - `--activation-checkpoint`: use activation checkpointing for checkpointed model forward segments when gradients are enabled and GPU memory is tight
  - optional train overrides: `--runs`, `--epochs`, `--patience`, `--lr`, `--batch-size`, `--train-ratio`, `--val-ratio`, `--seed-mode`, `--seed-base`, `--allow-duplicate-seeds`
  - optional routing fields: `--log-file`, `--tag-prefix`
- Output:
  - stdout success: `case_manifest` payload with `cases` and `summary.total`
  - stdout failure: `expand_cases_error` payload when `--output-format json`
  - exit code 0: expansion succeeded
  - exit code non-zero: one or more inputs were invalid
- Side effects:
  - none
- Use when:
  - you need a low-level planning artifact before execution
  - you need stable `case_id` values for later validation or scheduling
- Do not use when:
  - you only need to execute one already-known job

#### gplab-query
- Purpose: Read persisted JSONL experiment records and emit flat summaries or grouped benchmark reports.
- Command: `gplab-query --log-file <path> --output-format json`
- Inputs:
  - `--log-file`: JSONL record file
  - optional filters: `--pool`, `--dataset`, `--model-type`, `--tag`
  - optional report flags: `--report`, `--sort-by`, `--show-spec`, `--show-replay`
- Output:
  - stdout success: `query_result` or `query_report`
  - stdout failure: `query_error` payload when `--output-format json`
  - exit code 0: query succeeded
  - exit code non-zero: log read, validation, or filtering failed
- Side effects:
  - none
- Use when:
  - you need persisted benchmark summaries
  - you need to group records by the benchmark comparison boundary
- Do not use when:
  - you need runtime normalization of a job before execution

#### gplab-replay
- Purpose: Rebuild a strict in-memory job from one record, compare runtime metadata, and optionally rerun it.
- Command: `gplab-replay --log-file <path> --record-id <id> --output-format json`
- Inputs:
  - `--log-file`: JSONL record file
  - `--record-id`: record identifier
  - optional execution controls: `--replay-log-file`, `--run`
- Output:
  - stdout success: `replay_result`
  - stdout failure: `replay_error` payload when `--output-format json`
  - exit code 0: replay metadata generation succeeded and optional rerun succeeded
  - exit code non-zero: record lookup, job reconstruction, or optional rerun failed
- Side effects:
  - does not generate TOML configs
  - when `--run` is set, executes training in the current process
  - when `--replay-log-file` is set, rerun may append a new record
- Use when:
  - you need deterministic reconstruction of one stored record
  - you need runtime compatibility metadata
  - you need rerun handoff metadata such as rerun `record_id`
- Do not use when:
  - you need to compare arbitrary jobs that were never logged
  - you need to run broad validation over many cases

### Mutating Tools

#### gplab-run-job
- Purpose: Execute one complete normalized automation job.
- Command: `gplab-run-job --job-file <path> --output-format json`
- Inputs:
  - `--job-file`: path to one complete job JSON object
  - `--output-format`: `json` or `text`
- Output:
  - stdout success: `train_result`
  - stdout failure: `train_error` payload when `--output-format json`
  - exit code 0: training finished successfully
  - exit code non-zero: configuration or runtime failure
- Side effects:
  - runs training
  - may append to the job's `log_file` if one is set
- Preconditions:
  - normalize the job first if the caller needs explicit defaults or early validation
- Use when:
  - you need the strict automation execution mode
  - you need complete strict request semantics
- Do not use when:
  - you need only normalization
  - you need case expansion without execution

#### gplab-validate
- Purpose: Execute a caller-directed smoke validation set and return both plan and per-case results.
- Command: `gplab-validate --pools <csv> --datasets <csv> --output-format json`
- Inputs:
  - `--pools`: comma-separated pools
  - `--datasets`: comma-separated datasets
  - `--model-type`: one model variant
  - execution overrides: `--pool-ratio`, `--pool-nonlinearity`, `--activation-checkpoint`, `--runs`, `--epochs`, `--patience`, `--lr`, `--batch-size`, `--train-ratio`, `--val-ratio`
  - optional logging and seed controls: `--log-file`, `--seed-mode`, `--seed-base`, `--seed-list`, `--allow-duplicate-seeds`, `--tag-prefix`
- Output:
  - stdout success: `validation_result`
  - stdout failure: `validation_error` payload when `--output-format json`
  - exit code 0: all cases passed
  - exit code non-zero: one or more cases failed or orchestration failed
- Side effects:
  - executes each planned job in the current process
  - may append records if `--log-file` is set
  - does not generate config files
- Preconditions:
  - use caller-provided case scope; this tool does not choose validation policy for you
- Use when:
  - you need minimal end-to-end structural validation across selected cases
  - you need machine-readable per-case results
- Do not use when:
  - you need a scheduler, sweep engine, or change-impact analyzer
  - you only need to validate one job file; use `gplab-normalize-job` or `gplab-run-job` instead

## Rules
- Prefer `gplab-run-job` over `gplab-train` for automation execution.
- If the task only requires normalization or planning, do not call training or validation tools.
- Before executing any mutating tool, prefer `gplab-normalize-job` or `gplab-expand-cases` when the caller benefits from explicit request objects or case plans.
- Treat payload shapes documented in `Static Facts` as public interface contracts. Do not rely on undocumented top-level fields.
- When static facts conflict with runtime behavior, prefer code and runtime results, and report the conflict.
- Do not invent datasets, pools, output formats, or replay compatibility states beyond `compatible` and `mismatch`.
- `gplab-validate` is a thin orchestrator. Do not treat it as project-owned policy about which cases must be run.
- Do not modify this file's documented constants unless the user explicitly requests interface changes.

## Recommended Workflow
1. Read the user task and decide whether it requires normalization, planning, execution, replay, querying, or validation.
2. If the task starts from a job file, run `gplab-normalize-job` first when early schema validation matters.
3. If the task starts from combinations rather than one job, run `gplab-expand-cases` first to obtain a stable manifest.
4. Use `gplab-run-job` for strict single-job execution, `gplab-query` for persisted summaries, `gplab-replay` for deterministic reruns, and `gplab-validate` for caller-directed smoke checks.
5. After any mutating operation, prefer reading structured JSON payloads rather than scraping console text.

## Examples

### Good
- If the task is "turn these pool/dataset combinations into executable units", run `gplab-expand-cases` and stop.
- If the task is "execute this automation job file", run `gplab-normalize-job` if explicit validation is needed, then run `gplab-run-job`.
- If the task is "rerun record X and capture the new record id", run `gplab-replay --run --output-format json` and read `rerun.record_id`.

### Bad
- Do not use `gplab-validate` when the task only needs one strict job execution.
- Do not scrape human-readable stdout from machine-facing tools when `--output-format json` exists.
- Do not assume benchmark grouping includes pool choice.
