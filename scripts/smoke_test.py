"""Run built-in pool/dataset jobs in separate processes and save their JSON responses.

Environment overrides match the shell entrypoint: POOLS and DATASETS are
space-separated names; training fields use uppercase names (e.g. EPOCHS).
"""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from gplab.data import DATASET_PROFILES
from gplab.layers.pool import POOLING_PROFILES


def main() -> None:
    """Exercise the public job boundary and fail if any requested experiment fails."""
    python = shlex.split(os.environ.get("PYTHON_CMD", sys.executable))
    results = []
    for dataset in os.environ.get("DATASETS", " ".join(DATASET_PROFILES)).split():
        for pool in os.environ.get("POOLS", " ".join(POOLING_PROFILES)).split():
            job = {
                "experiment": {
                    "dataset": dataset,
                    "pool": {"name": pool, "params": {"ratio": float(os.environ.get("POOL_RATIO", "0.5"))}},
                    "model": {"variant": os.environ.get("MODEL_VARIANT", "sum")},
                    "training": {
                        **{field: int(os.environ.get(field.upper(), default)) for field, default in
                           (("num_runs", "1"), ("epochs", "1"), ("patience", "0"), ("batch_size", "16"))},
                        "lr": float(os.environ.get("LR", "0.0005")),
                        "split": {"train": float(os.environ.get("SPLIT_TRAIN", "0.8")),
                                  "val": float(os.environ.get("SPLIT_VAL", "0.1"))},
                        "seeds": {"mode": "auto", "base": int(os.environ.get("SEED_BASE", "20260320"))},
                    },
                },
                "log_file": os.environ.get("LOG_FILE") or None,
                "tag": f"{os.environ.get('TAG_PREFIX', 'smoke')}_{dataset}_{pool}",
            }
            completed = subprocess.run(
                [*python, "-m", "gplab.cli.run_train_job", "--job-stdin", "--output-format", "json"],
                input=json.dumps(job), text=True, stdout=subprocess.PIPE, check=False,
            )
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError:
                payload = {"ok": False, "error": "Process did not return JSON", "stdout": completed.stdout}
            results.append({"dataset": dataset, "pool": pool, "exit_code": completed.returncode,
                            "payload": payload})
    ok = all(result["exit_code"] == 0 and result["payload"].get("ok") for result in results)
    path = Path(os.environ.get("RESULTS_PATH", "/tmp/gplab_smoke_result.json"))
    path.write_text(json.dumps({"ok": ok, "results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(results)} smoke results to {path}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
