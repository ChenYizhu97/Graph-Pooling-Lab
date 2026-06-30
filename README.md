# Graph Pooling Lab (GPLab)

GPLab is a lightweight benchmark harness for graph pooling methods on graph
classification tasks. Its core abstraction is a `BenchmarkCase`: a dataset,
model, pooling method, and training protocol executed under one fixed benchmark
protocol.

Read [PROTOCOL.md](PROTOCOL.md) for the stable benchmark rules. Automation
clients should also read [AGENT_REFERENCE.md](AGENT_REFERENCE.md).

```mermaid
flowchart LR
    A["BenchmarkRequest"] --> B["PreparedRun"]
    B --> C["Shared Graph Classifier"]
    C --> D{"Pooling"}
    D --> E["Sparse Poolers"]
    D --> F["Dense Poolers via Adapter"]
    E --> G["Shared Downstream Conv + Readout"]
    F --> G
    G --> H["Multi-seed Train / Val / Test"]
    H --> I["ExperimentRecord"]
    I --> J["gplab-query"]
    I --> K["gplab-replay"]
```

## Scope

`BenchmarkRequest` is the executable request wrapper around a `BenchmarkCase`
and `ExecutionOptions`. `PreparedRun` adds the loaded dataset profile and
resolved `RunPlan` with concrete seeds and split indices.

GPLab currently targets:

- TU datasets only
- graph classification only
- one pooling stage per model
- one shared downstream path after pooling

It is a benchmark harness, not a general-purpose graph learning framework.

## Layout

```text
src/gplab/
  benchmark/      # BenchmarkCase, BenchmarkRequest, RunPlan, comparison keys
  cli/            # gplab-* entrypoints
  data/           # TU loading and split helpers
  experiment/     # execution, record, result assembly
  jobs/           # strict job schema and case manifests
  layers/         # conv/pool resolver and pooling adapters
  model/          # shared graph classifier backbone
```

## Install

```bash
conda activate torch_env
python3 -m pip install -e .
```

GPLab depends on PyTorch, PyG, Typer, Rich, TOML, NumPy, and tqdm.

## Quick Start

Run one human-oriented experiment:

```bash
gplab-train --pool sagpool --pool-ratio 0.5 --dataset PROTEINS
```

Use the plain model variant:

```bash
gplab-train --pool sagpool --pool-ratio 0.5 --dataset PROTEINS --model-variant plain
```

Append the record to a JSONL log:

```bash
gplab-train \
  --pool sparsepool \
  --pool-ratio 0.5 \
  --dataset PROTEINS \
  --log-file runs/bench.jsonl \
  --tag baseline_proteins
```

Replay an exact seed list:

```bash
gplab-train \
  --pool diffpool \
  --pool-ratio 0.5 \
  --dataset PROTEINS \
  --seed-mode list \
  --seed-list 101,202,303
```

## Strict Job JSON

Machine-facing execution uses a complete JSON object with `case` and
`execution` blocks:

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

Normalize without running:

```bash
gplab-normalize-job --job-file job.json --output-format json
```

Run the job:

```bash
gplab-run-job --job-file job.json --output-format json
```

## Automation Entrypoints

- `gplab-normalize-job`: validate one strict job.
- `gplab-expand-cases`: generate complete jobs from combinations.
- `gplab-run-job`: execute one strict job.
- `gplab-query`: summarize JSONL records.
- `gplab-replay`: rebuild and optionally rerun one record.
- `gplab-validate`: caller-directed smoke validation.

Generate a case manifest:

```bash
gplab-expand-cases \
  --pools sagpool,diffpool \
  --datasets MUTAG,PROTEINS \
  --model-variants sum,plain \
  --output-format json
```

Run smoke validation:

```bash
gplab-validate --pools sagpool,diffpool --datasets MUTAG,PROTEINS
```

## Supported Datasets

- `MUTAG`
- `PROTEINS`
- `ENZYMES`
- `FRANKENSTEIN`
- `Mutagenicity`
- `AIDS`
- `DD`
- `NCI1`
- `COX2`

## Pooling Methods

Built-in pools:

- `nopool`
- `topkpool`
- `sagpool`
- `asapool`
- `sparsepool`
- `mincutpool`
- `diffpool`
- `densepool`

Sparse poolers operate directly on sparse graph batches. Dense poolers are
wrapped by `DensePoolAdapter`, which converts sparse input batches to dense
tensors, applies dense pooling, and converts fixed cluster slots back to sparse
format for the shared downstream backbone.

## Custom Pooling Plugins

Custom pooling factories use:

```text
<python_module>:<factory_name>
```

Recommended factory signature:

```python
def build_pool(
    in_channels: int,
    ratio: float = 0.5,
    avg_node_num=None,
    nonlinearity="relu",
):
    ...
```

The factory must return a `torch.nn.Module`; the module must return
`PoolOutput` and implement `reset_parameters()`.

## Experiment Records

Each JSONL record contains:

- `case`: the benchmark case
- `execution`: execution-only options
- `run_plan`: resolved seeds and split indices
- `runtime`: environment metadata
- `result`: per-run and aggregate metrics
- `record_id`: content hash

Query records:

```bash
gplab-query --log-file runs/bench.jsonl --report
gplab-query --log-file runs/bench.jsonl --model-variant plain
gplab-query --log-file runs/bench.jsonl --show-case --show-replay
```

Replay one record:

```bash
gplab-replay --log-file runs/bench.jsonl --record-id <record_id>
gplab-replay --log-file runs/bench.jsonl --record-id <record_id> --run
```

## Configuration

`config/model.toml` controls model defaults.

`config/experiment.toml` contains:

- `[training]`
- `[training.split]`
- `[training.seeds]`
- `[execution]`

CLI flags override these defaults before building a `BenchmarkCase`.
