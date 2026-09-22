# Laya Local

在 Apple Silicon Mac 上本地运行 [Laya](https://huggingface.co/convaiinnovations/laya) typed-decision 模型。项目同时提供：

- 浏览器 Playground：`http://127.0.0.1:7860`
- REST API：`POST /predict`
- OpenAPI 文档：`http://127.0.0.1:7860/docs`
- 健康检查：`GET /health`

默认加载 `convaiinnovations/laya` 仓库中的 `multilingual` checkpoint，适合中文和其他非英语输入。设备设置为 `auto`，Apple Silicon 会优先使用 MPS，不可用时由 Laya 回退到 CPU。

## 首次安装

开始前需要准备 Git 和 [`uv`](https://docs.astral.sh/uv/getting-started/installation/)。`setup.sh` 会通过 `uv` 准备 Python，但不会安装 `uv` 本身。

```bash
git clone https://github.com/koljahuang/laya-local.git
cd laya-local
chmod +x scripts/setup.sh scripts/run.sh
./scripts/setup.sh
```

安装脚本会：

1. 通过 `uv` 准备 Python 3.12；
2. 按 `uv.lock` 安装依赖；
3. 将模型下载到项目内的 `.cache/huggingface/`；
4. 加载模型并执行一次中文推理。

模型和 PyTorch 会占用较多磁盘空间，首次执行时间取决于 Hugging Face 下载速度。

## 启动

```bash
./scripts/run.sh
```

看到 `Laya ready` 和 `Uvicorn running` 后，打开：

<http://127.0.0.1:7860>

服务固定使用一个 Uvicorn worker，避免重复加载模型；推理请求也会串行执行，防止多个请求同时抢占 Apple 的统一内存。

## 调用 API

```bash
curl -s http://127.0.0.1:7860/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {"message": "我被重复扣款，请退款。"},
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "哪个团队应该处理？",
        "criteria": {
          "billing": "账单、付款和退款",
          "technical": "故障和系统异常",
          "sales": "价格与合同"
        }
      },
      "refund_requested": {
        "type": "noul",
        "instructions": "客户是否明确要求退款？"
      }
    }
  }' | python3 -m json.tool
```

也可以在服务启动后运行自动冒烟测试：

```bash
uv run python scripts/smoke_test.py
```

## Amazon SageMaker AI

`sagemaker/` 提供实时端点所需的自定义容器适配：

- `serve.py`：监听 8080 端口，提供 `/ping` 和 `/invocations`；
- `Dockerfile`：构建 `linux/amd64` CPU 推理镜像；
- `prepare_model.py`：把 Hugging Face checkpoint 准备到容器构建目录；
- `validation-report.md`：记录 `ml.m5.xlarge` 端点的验证结果和 CloudWatch 指标。

模型权重、临时压缩包和本地缓存不会提交到 Git。部署过程与验证结果见 [Laya 部署实践](docs/aws-blog/laya-aws-blog.md)。

## 配置

通过环境变量覆盖默认配置：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LAYA_MODEL` | `convaiinnovations/laya` | Hugging Face 模型仓库 |
| `LAYA_SUBFOLDER` | `multilingual` | checkpoint 子目录 |
| `LAYA_DEVICE` | `auto` | `auto`、`mps` 或 `cpu` |
| `LAYA_HOST` | `127.0.0.1` | 监听地址，仅本机访问更安全 |
| `LAYA_PORT` | `7860` | 服务端口 |
| `HF_HOME` | 项目下 `.cache/huggingface` | 模型缓存目录 |

示例：强制使用 CPU：

```bash
LAYA_DEVICE=cpu ./scripts/run.sh
```

使用英语 checkpoint：

```bash
LAYA_SUBFOLDER='' ./scripts/run.sh
```

使用 typed-decisions checkpoint：

```bash
LAYA_SUBFOLDER=typed-decisions ./scripts/setup.sh
LAYA_SUBFOLDER=typed-decisions ./scripts/run.sh
```

## 常见问题

### 第一次启动很慢

首次安装需要下载 PyTorch 和模型权重。`setup.sh` 完成后，后续启动直接读取本地缓存。

### MPS 报不支持某个算子

启动脚本已设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`，不受支持的算子可以回退 CPU。如果仍不稳定，可执行：

```bash
LAYA_DEVICE=cpu ./scripts/run.sh
```

### 端口被占用

```bash
LAYA_PORT=7861 ./scripts/run.sh
```

### 想让局域网其他设备访问

默认只监听 `127.0.0.1`。确认本机防火墙和网络安全后，再显式执行：

```bash
LAYA_HOST=0.0.0.0 ./scripts/run.sh
```

服务没有认证，不要直接暴露到公网。

## 限制

- Laya 输出结构化判断而不是自然语言，不适合开放式问答和长链推理。
- Demo 的概率不能直接等同于业务可靠性；上线前应使用自己的数据评估准确率和校准情况。
- 当前本地服务面向单机和开发验证，没有鉴权、限流、持久队列与监控。
