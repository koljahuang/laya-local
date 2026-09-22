#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export LAYA_MODEL="${LAYA_MODEL:-convaiinnovations/laya}"
export LAYA_SUBFOLDER="${LAYA_SUBFOLDER-multilingual}"
export LAYA_DEVICE="${LAYA_DEVICE:-auto}"

printf '1/3 Installing Python 3.12 if needed...\n'
uv python install 3.12
printf '2/3 Installing locked dependencies...\n'
uv sync --python 3.12 --locked
printf '3/3 Downloading and validating the selected checkpoint...\n'
uv run python scripts/download_model.py

printf '\nSetup complete. Start Laya with:\n  %s/scripts/run.sh\n' "$ROOT"
