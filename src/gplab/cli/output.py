from contextlib import nullcontext, redirect_stdout
import json
import sys
from typing import Any, Optional

import typer


OUTPUT_FORMATS = ("text", "json")


def validate_output_format(value: str) -> str:
    if value not in OUTPUT_FORMATS:
        raise typer.BadParameter(
            f"output_format must be one of: {', '.join(OUTPUT_FORMATS)}.",
            param_hint="--output-format",
        )
    return value


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def redirect_stdout_for_json(enabled: bool):
    if enabled:
        return redirect_stdout(sys.stderr)
    return nullcontext()


def build_error_payload(kind: str, exc: Exception, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    error_type = "runtime_error"
    if isinstance(exc, typer.BadParameter) or isinstance(exc, ValueError):
        error_type = "config_error"
    elif isinstance(exc, FileNotFoundError):
        error_type = "file_not_found"

    payload: dict[str, Any] = {
        "ok": False,
        "kind": kind,
        "error": {
            "type": error_type,
            "message": str(exc),
        },
    }
    for key in ("field", "expected", "missing", "unknown"):
        value = getattr(exc, key, None)
        if value is not None:
            payload["error"][key] = value
    if details:
        payload["error"]["details"] = details
    return payload
