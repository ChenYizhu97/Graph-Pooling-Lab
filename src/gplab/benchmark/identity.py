"""Group completed records by the settings used for accuracy comparisons."""
import hashlib
import json


def _hash_payload(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def compute_comparison_group_key(record: dict) -> str:
    """Group pools using the same model, training budget, and actual seeds and splits.

    Seed-generation policies and split fractions do not distinguish completed
    experiments when they produced identical runs. Activation checkpointing is also
    excluded from accuracy grouping; memory/time comparisons must account for it.
    Equal seeds alone cannot establish that two records trained and tested on
    the same examples; their concrete splits must also match.
    """
    config = record["experiment"]
    training = {key: value for key, value in config["training"].items()
                if key not in {"seeds", "split", "activation_checkpoint"}}
    return _hash_payload({
        "dataset": config["dataset"],
        "input_type": config["input_type"],
        "model": config["model"],
        # Compression protocols are report cohorts, never a comparability predicate.
        "compression": config["compression"],
        "training": training,
        "runs": [{"seed": run["seed"], "split": run["split"]}
                 for run in record["result"]["runs"]],
    })
