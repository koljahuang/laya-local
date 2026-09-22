#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export LAYA_MODEL="${LAYA_MODEL:-convaiinnovations/laya}"
export LAYA_SUBFOLDER="${LAYA_SUBFOLDER:-multilingual}"
export LAYA_DEVICE="${LAYA_DEVICE:-auto}"

exec uv run uvicorn app:app \
  --host "${LAYA_HOST:-127.0.0.1}" \
  --port "${LAYA_PORT:-7860}" \
  --workers 1
