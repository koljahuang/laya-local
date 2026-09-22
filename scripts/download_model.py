from __future__ import annotations

import json
import os
import time

import laya  # pyright: ignore[reportMissingImports]  # installed in the project uv environment

model_id = os.getenv("LAYA_MODEL", "convaiinnovations/laya")
subfolder = os.getenv("LAYA_SUBFOLDER", "multilingual").strip() or None
requested_device = os.getenv("LAYA_DEVICE", "auto").strip().lower()
device = None if requested_device == "auto" else requested_device

print(f"Loading {model_id}/{subfolder or 'root'} (device={requested_device})...", flush=True)
started = time.perf_counter()
agent = laya.load(model_id, subfolder=subfolder, device=device)
load_seconds = time.perf_counter() - started

state = {"message": "我被重复扣款，请尽快退款，否则我会取消服务。"}
questions = {
    "department": {
        "type": "choice",
        "instructions": "哪个团队应该处理这条消息？",
        "criteria": {
            "billing": "账单、付款和退款",
            "technical": "故障和系统异常",
            "sales": "价格与合同",
        },
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "客户是否明确要求退款？",
    },
}

started = time.perf_counter()
result = agent.predict(state, questions)
latency_ms = (time.perf_counter() - started) * 1000

print(
    json.dumps(
        {
            "status": "ready",
            "model": model_id,
            "subfolder": subfolder,
            "device": str(agent.device),
            "load_seconds": round(load_seconds, 2),
            "first_inference_ms": round(latency_ms, 2),
            "answers": result["answers"],
        },
        ensure_ascii=False,
        indent=2,
    )
)
