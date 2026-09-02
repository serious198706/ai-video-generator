# Lambda 接单层

Java 调 API Gateway HTTP API，再进这支 Lambda。只校验 URL、写 Redis，**不下图、不推理**。GPU worker `LPOP` 后再下载。

必须和 ElastiCache **同一 VPC**（Security Group 放行 Redis）。Serverless 用 `rediss://`。

## 打包

依赖里的 `pydantic_core` 是二进制扩展。在 Mac 上直接 `pip install -t` 会打进 darwin/arm64 的 `.so`，Lambda（Linux x86_64）会报：

`No module named 'pydantic_core._pydantic_core'`

必须交叉装 **manylinux x86_64 / CPython 3.12** 的 wheel（不要把 `infer/`、torch 打进 zip）：

```bash
cd server
chmod +x lambda_api/package.sh
./lambda_api/package.sh /tmp/wan22-lambda.zip
```

等价手写：

```bash
python3 -m pip install -r lambda_api/requirements-lambda.txt -t /tmp/wan22-lambda-pkg \
  --python-version 3.12 \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --only-binary=:all:
# 再 rsync wan22/ + lambda_api/，删 infer、media、worker.py、api/app.py，zip
```

Lambda Runtime 选 **Python 3.12**、架构 **x86_64**。Handler：`lambda_api.handler.handler`。

## 环境变量

| 变量 | 说明 |
| --- | --- |
| `WAN22_REDIS_URL` | `rediss://…` ElastiCache Serverless |
| `WAN22_QUEUE_MAX` | 默认 500。`LLEN({wan22}:queue) >=` 此值则 429 |
| `WAN22_IMAGE_HOSTS` | 图片 URL 白名单。空则不限制 host（仍要求 https、公网） |
| `WAN22_WEBHOOK_HOSTS` | webhook URL 白名单。空则不限制 host（仍要求 https） |
| `WAN22_DEFAULT_PROMPT` / `WAN22_NEGATIVE` | 可选，与 GPU 保持一致 |

不需要 `WAN22_S3_*`、模型路径、Foley。

## API Gateway

HTTP API → Lambda 代理。Java Base URL 换成 API Gateway 域名（不要再打 GPU `:8000`）。

本机联调（需本机或隧道能连 Redis）：

```bash
WAN22_REDIS_URL=redis://127.0.0.1:6379/0 \
WAN22_IMAGE_HOSTS=cdn.example.com \
WAN22_WEBHOOK_HOSTS=api.example.com \
uvicorn lambda_api.app:app --port 8000
```
