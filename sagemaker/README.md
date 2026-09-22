# 部署到 Amazon SageMaker AI

本目录提供可执行的 SageMaker AI 实时端点部署流程。`manage.sh` 会完成以下工作：

1. 下载并整理 Laya multilingual checkpoint；
2. 构建 `linux/amd64` CPU 推理镜像；
3. 创建或复用 Amazon ECR 仓库并推送镜像；
4. 创建 SageMaker AI Model、Endpoint Configuration 和实时 Endpoint；
5. 等待 Endpoint 进入 `InService`；
6. 保存资源名称，供调用、查看日志和清理使用。

模型权重会放入镜像的 `/opt/ml/model`，容器启动后不需要访问 Hugging Face。

## 前提条件

本机需要：

- AWS CLI v2；
- Docker，并启用 `docker buildx`；
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)；
- 有权创建 SageMaker AI 与 ECR 资源的 AWS 身份；
- 一个允许 `sagemaker.amazonaws.com` 代入的 SageMaker 执行角色。

部署调用方至少需要 ECR push、SageMaker Model/Endpoint 管理、`iam:PassRole` 和 `sts:GetCallerIdentity` 权限。SageMaker 执行角色需要读取同一账户中的 ECR 镜像；可为该角色附加 `AmazonEC2ContainerRegistryReadOnly`，并按企业安全要求进一步收紧权限。

如果使用 AWS IAM Identity Center 配置的 CLI profile，先登录：

```bash
export AWS_PROFILE=<profile-name>
aws sso login --profile "$AWS_PROFILE"
```

## 创建执行角色（没有现成角色时）

如果账户里已有 SageMaker 执行角色，可以跳过本节。否则，具有 IAM 管理权限的身份可以创建一个只供此示例使用的角色：

```bash
export ROLE_NAME=laya-sagemaker-execution

cat >/tmp/laya-sagemaker-trust.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "sagemaker.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
JSON

aws iam create-role \
  --role-name "$ROLE_NAME" \
  --assume-role-policy-document file:///tmp/laya-sagemaker-trust.json

aws iam attach-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

export SAGEMAKER_ROLE_ARN="$(aws iam get-role \
  --role-name "$ROLE_NAME" \
  --query 'Role.Arn' \
  --output text)"
```

`manage.sh` 不会修改或删除这个角色。部署调用方仍然需要对该角色拥有 `iam:PassRole`。

## 一次完成部署

在仓库根目录执行：

```bash
export AWS_PROFILE=<profile-name>                 # 使用默认凭证链时可省略
export AWS_REGION=us-west-2
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account-id>:role/<sagemaker-role>

chmod +x sagemaker/manage.sh
./sagemaker/manage.sh deploy
```

默认实例为 `ml.m5.xlarge`。可以在部署前覆盖：

```bash
export INSTANCE_TYPE=ml.m5.xlarge
export ECR_REPOSITORY=laya-sagemaker
export NAME_PREFIX=laya
```

为便于复现，生产构建应固定 Hugging Face revision：

```bash
export LAYA_MODEL_REVISION=<hugging-face-commit-sha>
./sagemaker/manage.sh deploy
```

部署期间会下载模型、构建约 1 GB 的压缩镜像并等待 Endpoint 就绪，通常需要数分钟。中断本地等待不会自动删除云端资源；可以随后运行 `status` 或 `cleanup`。

## 查看状态

```bash
./sagemaker/manage.sh status
```

资源名称和镜像信息保存在本地 `sagemaker/.deployment.env`。该文件可能包含 AWS 账户 ID，因此已被 Git 忽略，不应手动提交。

## 调用 Endpoint

仓库提供了与文章一致的中文请求：

```bash
./sagemaker/manage.sh invoke
```

默认读取 `sagemaker/request.json`，并将响应保存到 `/tmp/laya-sagemaker-response.json`。也可以提供其他文件：

```bash
./sagemaker/manage.sh invoke ./my-request.json ./response.json
```

脚本最终调用的 AWS API 等价于：

```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name <endpoint-name> \
  --content-type application/json \
  --body fileb://sagemaker/request.json \
  --cli-binary-format raw-in-base64-out \
  response.json
```

## 查看容器日志

```bash
./sagemaker/manage.sh logs       # 默认从最近 1 小时开始并持续跟踪
./sagemaker/manage.sh logs 10m   # 从最近 10 分钟开始
```

对应的 CloudWatch Logs 日志组为：

```text
/aws/sagemaker/Endpoints/<endpoint-name>
```

## 删除资源

实时 Endpoint 会按实例运行时间计费。验证结束后应及时执行：

```bash
./sagemaker/manage.sh cleanup
```

这会删除 Endpoint、Endpoint Configuration 和 Model，默认保留 ECR 仓库。若确认仓库中的镜像也不再需要：

```bash
DELETE_ECR_REPOSITORY=1 ./sagemaker/manage.sh cleanup
```

脚本不会删除你提供的 SageMaker 执行角色。如果上面创建的角色只用于本次验证，可以在 Endpoint、Endpoint Configuration 和 Model 删除后再清理：

```bash
aws iam detach-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam delete-role --role-name "$ROLE_NAME"
```

## 单独构建和本地测试容器

如需在部署前检查容器：

```bash
uv run python sagemaker/prepare_model.py

docker buildx build \
  --platform linux/amd64 \
  --load \
  -f sagemaker/Dockerfile \
  -t laya-sagemaker:local .

docker run --rm --platform linux/amd64 -p 8080:8080 laya-sagemaker:local
```

另开终端检查：

```bash
curl -s http://127.0.0.1:8080/ping | python3 -m json.tool

curl -s http://127.0.0.1:8080/invocations \
  -H 'Content-Type: application/json' \
  --data-binary @sagemaker/request.json | python3 -m json.tool
```

## 常见问题

### `AccessDenied` 或 `iam:PassRole`

运行部署的 AWS 身份必须有权把 `SAGEMAKER_ROLE_ARN` 传给 SageMaker AI。执行角色本身还必须信任 `sagemaker.amazonaws.com`。

### Endpoint 进入 `Failed`

先查看状态和日志：

```bash
./sagemaker/manage.sh status
./sagemaker/manage.sh logs
```

常见原因包括执行角色无法读取 ECR、镜像架构不是 `linux/amd64`、模型文件没有复制到 `/opt/ml/model`，或实例内存不足。

### 本地已有部署状态

为防止覆盖仍在计费的 Endpoint，检测到 `sagemaker/.deployment.env` 时脚本不会再次部署。先运行 `status` 确认资源，再执行 `cleanup`。
