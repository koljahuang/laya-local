from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import laya
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

MODEL_PATH = os.getenv("LAYA_MODEL_PATH", "/opt/ml/model")
DEVICE = os.getenv("LAYA_DEVICE", "cpu")

_agent: laya.Agent | None = None
_load_seconds: float | None = None
_inference_lock = threading.Lock()


def _status() -> dict[str, Any]:
    return {
        "status": "ready" if _agent is not None else "loading",
        "model_path": MODEL_PATH,
        "device": str(_agent.device) if _agent is not None else DEVICE,
        "load_seconds": _load_seconds,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _agent, _load_seconds
    started = time.perf_counter()
    agent = laya.load(MODEL_PATH, device=DEVICE)
    _agent = agent
    _load_seconds = round(time.perf_counter() - started, 3)
    print(
        f"Laya SageMaker container ready on {agent.device} in {_load_seconds}s",
        flush=True,
    )
    yield
    _agent = None


app = FastAPI(
    title="Laya SageMaker Inference",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


@app.get("/ping")
def ping() -> JSONResponse:
    status_code = 200 if _agent is not None else 503
    return JSONResponse(_status(), status_code=status_code)


@app.get("/health")
def health() -> JSONResponse:
    return ping()


@app.post("/invocations")
async def invocations(request: Request) -> JSONResponse:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    content_type = request.headers.get("content-type", "").split(";", 1)[0]
    if content_type != "application/json":
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Request body is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    questions = payload.get("questions")
    if "state" not in payload or not isinstance(questions, dict) or not questions:
        raise HTTPException(status_code=400, detail="Request requires state and non-empty questions")

    started = time.perf_counter()
    try:
        with _inference_lock:
            result = _agent.predict(payload["state"], questions)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}") from exc

    return JSONResponse(
        jsonable_encoder(
            {
                "result": result,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "deployment": _status(),
            }
        )
    )


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    return await invocations(request)


if __name__ == "__main__":
    bind_host = os.getenv("SAGEMAKER_BIND_HOST", "127.0.0.1")
    uvicorn.run(app, host=bind_host, port=8080, workers=1, access_log=True)
