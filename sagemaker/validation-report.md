# Laya SageMaker AI Validation Report

- Date: 2026-09-22
- Region: `us-west-2`
- Instance: `ml.m5.xlarge` (1 real-time endpoint instance)
- Cleanup policy: delete all temporary AWS resources after validation

## Container

- Platform: `linux/amd64`
- Python: 3.12
- PyTorch: 2.14.0 CPU
- Laya: 0.3.5
- Checkpoint: multilingual, copied to `/opt/ml/model`
- Protocol: `GET /ping`, `POST /invocations`, port 8080
- Local image size: 1,918,068,083 bytes
- ECR compressed image size: 962,942,242 bytes
- ECR digest: `sha256:ee74b44afa82b546e189334d151c224dcf297baeedf91492d2bf2cc8df15b851`

## Pre-deployment checks

The adapter was tested directly on macOS CPU and then from the exact `linux/amd64` image under local emulation.

| Test | Load time | Inference time | Result |
|---|---:|---:|---|
| Direct Python service, CPU | 25.525 s | 49.86 ms | HTTP 200, `billing` |
| `linux/amd64` Docker image under emulation | 206.71 s | 427.36 ms | HTTP 200, `billing` |

The emulated Docker timing is not representative of native SageMaker performance.

## SageMaker AI endpoint

- Endpoint state: `Creating` → `InService`
- Creation to `InService`: approximately 184 seconds
- Container model load in CloudWatch Logs: 28.154 seconds
- `/ping`: repeated HTTP 200 health checks
- `/invocations`: three HTTP 200 responses

### Invocation results

| Call | AWS CLI wall time | Container processing `latency_ms` | Department | Department probability | Urgency | Churn risk | Refund requested |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 2955.87 ms | 428.99 ms | billing | 0.9938 | 1.3578 | 0.0654 | 0.9945 |
| 2 | 2085.90 ms | 421.94 ms | billing | 0.9938 | 1.3578 | 0.0654 | 0.9945 |
| 3 | 2132.62 ms | 427.02 ms | billing | 0.9938 | 1.3578 | 0.0654 | 0.9945 |

The AWS CLI wall time includes local process startup, credential resolution, network transit, SageMaker overhead, and model execution. It is not a model latency benchmark.

### CloudWatch evidence

- `ModelLatency`: 436.55 ms
- `OverheadLatency`: 228.28 ms
- Endpoint log stream created successfully
- No `ERROR` or traceback was observed

CloudWatch metrics are published asynchronously. At query time, the available invocation metric had not yet aggregated all three log-confirmed calls.

## Cleanup

Deleted and verified absent:

- SageMaker Endpoint
- SageMaker Endpoint Configuration
- SageMaker Model
- ECR repository and image
- CloudWatch endpoint log group
- Temporary SageMaker IAM execution role

No S3 bucket or object was created. Total validation and cleanup wall-clock time was 1,671 seconds (27 minutes 51 seconds).
