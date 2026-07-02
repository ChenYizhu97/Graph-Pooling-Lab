from __future__ import annotations

from dataclasses import dataclass
import shlex
from typing import Optional

from gplab.benchmark.comparison import compute_record_benchmark_key
from gplab.experiment.record import summarize_record
from gplab.utils.validation import (
    validate_dataset_value,
    validate_model_variant_value,
    validate_pool_value,
)


SORT_FIELDS = ("mean", "std", "avg_best_epoch", "avg_val_loss")
LOWER_IS_BETTER = {"std", "avg_val_loss", "avg_best_epoch"}


class QuerySpecError(ValueError):
    def __init__(self, message: str, *, field: str | None = None, expected: str | None = None) -> None:
        super().__init__(message)
        self.field = field
        self.expected = expected


@dataclass(frozen=True)
class QuerySpec:
    log_file: str
    dataset: Optional[str] = None
    pool: Optional[str] = None
    model_variant: Optional[str] = None
    tag: Optional[str] = None
    sort_by: str = "mean"
    show_case: bool = False
    show_replay: bool = False

    def filters(self) -> dict:
        return {
            key: value
            for key, value in {
                "dataset": self.dataset,
                "pool": self.pool,
                "model_variant": self.model_variant,
                "tag": self.tag,
            }.items()
            if value is not None
        }


def validate_query_spec(spec: QuerySpec) -> None:
    if spec.sort_by not in SORT_FIELDS:
        raise QuerySpecError(
            f"sort_by must be one of: {', '.join(SORT_FIELDS)}.",
            field="sort_by",
            expected=f"one of: {', '.join(SORT_FIELDS)}",
        )
    if spec.dataset is not None:
        try:
            validate_dataset_value(spec.dataset)
        except ValueError as exc:
            raise QuerySpecError(str(exc), field="dataset") from exc
    if spec.pool is not None:
        try:
            validate_pool_value(spec.pool)
        except ValueError as exc:
            raise QuerySpecError(str(exc), field="pool") from exc
    if spec.model_variant is not None:
        try:
            validate_model_variant_value(spec.model_variant)
        except ValueError as exc:
            raise QuerySpecError(str(exc), field="model_variant") from exc


def select_records(records: list[dict], spec: QuerySpec) -> list[dict]:
    validate_query_spec(spec)
    selected = []
    for record in records:
        case = record["case"]
        if spec.dataset is not None and case["dataset"].lower() != spec.dataset.lower():
            continue
        if spec.pool is not None and case["pool"]["name"] != spec.pool:
            continue
        if spec.tag is not None and record["execution"].get("tag") != spec.tag:
            continue
        if spec.model_variant is not None and case["model"]["variant"] != spec.model_variant:
            continue
        selected.append(record)
    return selected


def _context(spec: QuerySpec, *, total_records: int, matched_records: int) -> dict:
    return {
        "source": "record_log",
        "log_file": spec.log_file,
        "filters": spec.filters(),
        "sort_by": spec.sort_by,
        "total_records": total_records,
        "matched_records": matched_records,
    }


def _summary_for_query(record: dict, spec: QuerySpec) -> dict:
    summary = summarize_record(record)
    if spec.show_case:
        summary["case"] = record["case"]
    if spec.show_replay:
        summary["replay_command"] = (
            f"gplab-replay --log-file {shlex.quote(spec.log_file)} --record-id {record['record_id']}"
        )
    return summary


def build_query_result(records: list[dict], spec: QuerySpec) -> dict:
    selected = _sort_records(select_records(records, spec), spec.sort_by)
    return {
        "ok": True,
        "kind": "query_result",
        "context": _context(spec, total_records=len(records), matched_records=len(selected)),
        "summaries": [_summary_for_query(record, spec) for record in selected],
    }


def _sort_value(record: dict, sort_by: str) -> float:
    return float(summarize_record(record)[sort_by])


def _sort_records(records: list[dict], sort_by: str) -> list[dict]:
    return sorted(
        records,
        key=lambda record: _sort_value(record, sort_by),
        reverse=sort_by not in LOWER_IS_BETTER,
    )


def _rank_groups(records: list[dict], sort_by: str) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(compute_record_benchmark_key(record), []).append(record)
    return [
        (benchmark_key, _sort_records(group, sort_by))
        for benchmark_key, group in groups.items()
    ]


def build_benchmark_report(records: list[dict], spec: QuerySpec) -> dict:
    selected = select_records(records, spec)
    groups = []
    for benchmark_key, ranked in _rank_groups(selected, spec.sort_by):
        first = ranked[0]
        tags = sorted(
            {
                record["execution"].get("tag")
                for record in ranked
                if record["execution"].get("tag") is not None
            }
        )
        summaries = []
        for index, record in enumerate(ranked, start=1):
            summary = _summary_for_query(record, spec)
            summary["rank"] = index
            summaries.append(summary)

        group = {
            "benchmark_key": benchmark_key,
            "comparison": {
                "dataset": first["case"]["dataset"],
                "model_variant": first["case"]["model"]["variant"],
                "pool_ratio": first["case"]["pool"]["ratio"],
                "pool_nonlinearity": first["case"]["pool"]["nonlinearity"],
            },
            "summaries": summaries,
        }
        if tags:
            group["tags"] = tags
        groups.append(group)

    return {
        "ok": True,
        "kind": "query_report",
        "context": _context(spec, total_records=len(records), matched_records=len(selected)),
        "groups": groups,
    }


def format_query_text(payload: dict) -> str:
    return "\n".join(str(summary) for summary in payload["summaries"])


def format_report_text(payload: dict) -> str:
    lines = []
    for group in payload["groups"]:
        comparison = group["comparison"]
        header_parts = [
            f"dataset={comparison['dataset']}",
            f"model={comparison['model_variant']}",
            f"ratio={comparison['pool_ratio']}",
            f"pool_nonlinearity={comparison['pool_nonlinearity']}",
            f"benchmark={group['benchmark_key']}",
        ]
        tags = group.get("tags", [])
        if len(tags) == 1:
            header_parts.append(f"tag={tags[0]}")
        elif len(tags) > 1:
            header_parts.append(f"tags={len(tags)}")
        lines.append(" | ".join(header_parts))

        for summary in group["summaries"]:
            corr = summary["val_loss_test_acc_corr"]
            corr_text = "n/a" if corr is None else f"{corr:.4f}"
            lines.append(
                f"{summary['rank']}. pool={summary['pool']} ratio={summary['pool_ratio']} "
                f"mean={summary['mean']:.4f} std={summary['std']:.4f} "
                f"avg_epoch={summary['avg_best_epoch']:.1f} avg_val_loss={summary['avg_val_loss']:.6f} "
                f"val_test_corr={corr_text} record_id={summary['record_id']}"
            )
        lines.append(
            "Interpretation: compare mean first, then use std for stability, avg_epoch for early-stop behavior, "
            "and val_test_corr to judge whether lower validation loss aligned with better test accuracy."
        )
        lines.append("")
    return "\n".join(lines).rstrip()
