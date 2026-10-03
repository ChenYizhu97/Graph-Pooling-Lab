"""Identify requested experiments and group completed records by their actual protocol."""
import hashlib
import json

from .config import ExperimentConfig


def _hash_payload(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def compute_experiment_id(config: ExperimentConfig) -> str:
    """Identify the full requested configuration, including pool and seed policy."""
    return _hash_payload(config.to_mapping())


def compute_record_benchmark_key(record: dict) -> str:
    """Group pools using the same model, training budget, and actual seeds and splits.

    Seed-generation policies and split fractions do not distinguish completed
    experiments when they produced identical runs. Concrete splits do: equal seeds
    alone cannot establish that two records trained and tested on the same examples.
    """
    config = record["experiment"]
    training = {key: value for key, value in config["training"].items()
                if key not in {"seeds", "split"}}
    return _hash_payload({
        "dataset": config["dataset"],
        "model": config["model"],
        "pool_protocol": {key: config["pool"][key] for key in ("ratio", "nonlinearity")},
        "training": training,
        "runs": [{"seed": run["seed"], "split": run["split"]}
                 for run in record["result"]["runs"]],
    })
