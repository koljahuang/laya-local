from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
DESTINATION = Path(os.getenv("LAYA_MODEL_DESTINATION", str(ROOT / "model"))).expanduser().resolve()
MODEL_ID = os.getenv("LAYA_MODEL", "convaiinnovations/laya")
SUBFOLDER = os.getenv("LAYA_SUBFOLDER", "multilingual").strip()
REVISION = os.getenv("LAYA_MODEL_REVISION", "").strip() or None
LOCAL_FILES_ONLY = os.getenv("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes"}

prefix = f"{SUBFOLDER}/" if SUBFOLDER else ""
snapshot = Path(
    snapshot_download(
        MODEL_ID,
        revision=REVISION,
        allow_patterns=[
            prefix + "rl_agent_config.json",
            prefix + "model.safetensors",
            prefix + "tokenizer/*",
            prefix + "encoder/*",
        ],
        local_files_only=LOCAL_FILES_ONLY,
    )
)
source = snapshot / SUBFOLDER if SUBFOLDER else snapshot
if not source.is_dir():
    raise SystemExit(f"Checkpoint directory not found: {source}")

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
