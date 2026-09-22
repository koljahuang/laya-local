from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "model"

snapshot = Path(
    snapshot_download(
        "convaiinnovations/laya",
        allow_patterns=[
            "multilingual/rl_agent_config.json",
            "multilingual/model.safetensors",
            "multilingual/tokenizer/*",
            "multilingual/encoder/*",
        ],
        local_files_only=True,
    )
)
source = snapshot / "multilingual"
if not source.is_dir():
    raise SystemExit(f"Multilingual checkpoint not found in local cache: {source}")

try:
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    shutil.copytree(source, DESTINATION, symlinks=False)
except OSError as exc:
    raise SystemExit(f"Could not prepare model directory: {exc}") from exc

required = [DESTINATION / "rl_agent_config.json", DESTINATION / "model.safetensors"]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"Prepared model is missing required files: {missing}")

size_bytes = sum(path.stat().st_size for path in DESTINATION.rglob("*") if path.is_file())
print(f"Prepared {DESTINATION} ({size_bytes / 1024 / 1024:.1f} MiB)")
