#!/usr/bin/env bash
# Local smoke test of the REAL trainer on a tiny subset. Never a real run.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src ./.venv/bin/python -m jaundice.train.train --config configs/smoke.yaml --tag smoke
