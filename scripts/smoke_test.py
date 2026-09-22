from __future__ import annotations

import http.client
import json
import os
from urllib.parse import urlparse

base_url = os.getenv("LAYA_URL", "http://127.0.0.1:7860")
parsed_url = urlparse(base_url)
if parsed_url.scheme != "http" or not parsed_url.hostname:
    raise SystemExit("LAYA_URL must be a local HTTP URL, for example http://127.0.0.1:7860")
payload = {
    "state": {"message": "我被重复扣款，请退款。"},
    "questions": {
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
    },
}

body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
connection = http.client.HTTPConnection(parsed_url.hostname, parsed_url.port or 80, timeout=120)
try:
    connection.request("POST", "/predict", body=body, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        raise SystemExit(f"Smoke test failed: HTTP {response.status} {response.read().decode('utf-8')}")
    result = json.loads(response.read().decode("utf-8"))
finally:
    connection.close()

print(
    json.dumps(
        {
            "http": 200,
            "latency_ms": result["latency_ms"],
            "device": result["deployment"]["actual_device"],
            "answers": result["result"]["answers"],
        },
        ensure_ascii=False,
        indent=2,
    )
)
