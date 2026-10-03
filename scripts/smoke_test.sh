#!/usr/bin/env bash
# Run isolated one-epoch jobs through the supported Job JSON entrypoint.
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec uv run --locked python scripts/smoke_test.py
