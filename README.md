# Wan 2.2 Adapter API

给 Java 后端调用的图生视频服务，协议对齐 a2e / Pixverse adapter。

Java 打 **API Gateway + Lambda** 接单；GPU 只做推理 worker。队列是 **ElastiCache Redis**（List 排队 + String 存任务/结果）。推理仍是 WAMU Lightning I2V，画布约 480×832，成片上传到 S3（可用 CloudFront 域名回传 `video_url`）。

```
server/
  start.sh deploy.sh requirements.txt .env.example README.md JAVA.md
  lambda_api/   API Gateway → Lambda（校验 URL、入队、查询；不下图）
  wan22/
    config.py
    log.py        按日滚动的 wan22.log
    api/          GPU 本机 /health /ready（不给 Java 接单）
    queue/        Redis 任务与 GPU worker
    infer/        Wan pipeline
    media/        下图、上传、webhook
    net/          URL 白名单 / SSRF
```

## 协议

Java 调用端见 [JAVA.md](JAVA.md)（提交 / 查询 / webhook / 探活）。Lambda 部署见 [lambda_api/README.md](lambda_api/README.md)。

`POST /v1/generate`，`application/json`，无鉴权。立刻 `202`（**不下图**，只校验 HTTPS/白名单并 `RPUSH`）：

```json
{ "id": "…", "task_id": "…", "status": "queued" }
```

```json
{
  "image": "https://dxxx.cloudfront.net/a.jpg",
  "prompt": "nsfwsks, she slowly turns her head",
  "negativePrompt": "blurry",
  "duration": 5,
  "resolution": "540p",
  "webhookUrl": "https://api.example.com/internal/wan/callback",
  "steps": 6,
  "quality": 6,
  "seed": 123,
  "lastImage": null,
  "audio": true
}
```

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `image` | 是 | HTTPS 图片 URL（S3 / CloudFront）；Lambda 只校验，GPU `LPOP` 后下载 |
| `prompt` | 否 | 空则用 `WAN22_DEFAULT_PROMPT` |
| `negativePrompt` | 否 | 会传给 pipeline；`guidance_scale=1` 时不生效 |
| `duration` | 否 | `(0, 15]` 秒，默认 5 |
| `resolution` | 否 | `540p` / `720p` / `1080p`，本轮只记录，画布仍 480×832 |
| `webhookUrl` | 否 | 成功和失败都会 POST |
| `steps` / `quality` / `seed` / `lastImage` | 否 | Wan 私有字段；`quality` 为导出 1–10 |
| `audio` | 否 | 是否配 Foley，默认 `true` |

`GET /v1/tasks/{id}` 读 Redis String。Lambda `/health` Ping Redis，不再因「模型未就绪」503。GPU 本机 `/ready` 仅运维用。`/docs` 默认关闭。

图片 URL 必须解析到公网（防 SSRF）。Webhook 走独立白名单，**允许内网 IP**（Java 就在内网），但仍拒绝链路本地 / `169.254.169.254`。

Webhook body 与任务查询字段一致：`id`、`task_id`、`status`、`video_url`、`error`、`seed`、`duration`、`resolution`。失败时 `error` 仅为短码：`generate_failed` / `foley_failed` / `upload_failed` / `download_failed` / `interrupted`。配置了 `WAN22_WEBHOOK_SECRET` 时带 `X-Wan-Signature: sha256=…`。

## 配置

复制 `.env.example` 为 `.env`（不入库）。GPU `./start.sh` 会 source 它。

必填：

- `WAN22_REDIS_URL`（ElastiCache Serverless 用 `rediss://`）
- `WAN22_IMAGE_HOSTS`（空则拒绝图片 URL）
- `WAN22_WEBHOOK_HOSTS`（请求带了 `webhookUrl` 但此项为空则 400）
- `WAN22_S3_BUCKET`（GPU 非 DRY_RUN 启动时必填）

列表用逗号分隔，`.cloudfront.net` 这种写法按后缀匹配。

成片：`WAN22_S3_BUCKET` / `WAN22_S3_REGION` / `WAN22_S3_PREFIX`，回传 URL 用 `WAN22_S3_PUBLIC_BASE_URL`（CloudFront）。凭证走 GPU 机 IAM Role，或标准的 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`。

Lambda 与 GPU 共用 Redis 和白名单；Lambda **必须进 ElastiCache 同一 VPC**。Java Base URL 换成 API Gateway。

## 启动

GPU 机只跑 worker（本机 `127.0.0.1:8000` 的 `/health` `/ready` 给运维）：

```bash
cp .env.example .env
# 改 Redis / hosts / S3
./deploy.sh
./start.sh
```

`deploy.sh` 会装 Wan venv、WAMU 权重，以及独立的 Foley venv（`/opt/foley-venv`）和 HunyuanVideo-Foley XL 权重。Foley 沿用已有解释器，不重建 venv；3.14 上会改用带轮子的 Pillow / NumPy，避免源码编译。只要 Foley 的 python 在，`start.sh` 默认打开配音。不要 Foley：`.env` 里 `WAN22_FOLEY_ENABLE=0`，或 `WAN22_FOLEY_SKIP=1 ./deploy.sh`。

无 GPU 联调接口（本机 Redis）：

```bash
export WAN22_REDIS_URL=redis://127.0.0.1:6379/0
export WAN22_IMAGE_HOSTS=127.0.0.1,localhost
export WAN22_WEBHOOK_HOSTS=127.0.0.1,localhost
uvicorn lambda_api.app:app --port 8000
```

GPU dry-run worker：

```bash
export WAN22_DRY_RUN=1
export WAN22_REDIS_URL=redis://127.0.0.1:6379/0
uvicorn wan22.api.app:app --port 8001
```

`DRY_RUN` 写占位 mp4，不上传 S3，`video_url` 为 `http://127.0.0.1/dry-run/{id}.mp4`。

## 日志

写在 `WAN22_LOG_DIR`（默认 `./logs`）：

- 当前：`logs/wan22.log`
- 按日滚动：`logs/wan22.log.YYYY-MM-DD`（本地时区午夜，默认保留 30 天）
- 同时打到 stdout，方便 journald / 终端

记录入队、拒单（400/429/503）、推理开始/结束（含 seed、分辨率、耗时）、上传 URL、webhook 成败、进程重启后重试。uvicorn access 也进同一文件。

## 队列

ElastiCache Redis，哈希标签 `{wan22}`：

- `{wan22}:queue` List：入队 `RPUSH`，GPU `LPOP` 抢任务（不用 BRPOP，Serverless TLS 会掐阻塞读）
- `{wan22}:task:{id}` String：整份任务 JSON；查询、成功结果、失败状态都更新这把 key

`LLEN(queue) >= WAN22_QUEUE_MAX`（默认 **500**）返回 429；正在跑的不算进这 500。失败最多重试 3 次（含首次）：未超限则 `RPUSH` 回去；`attempts >= 3` 则 `failed` + webhook。进程崩溃只重入本机记下的 `running`，不抢别的机器的任务。

## 音频（可选）

`deploy.sh` 装好后，worker 在成片上传前：把 Wan 挪到 CPU → HunyuanVideo-Foley XL 看视频出 wav（固定 `WAN22_FOLEY_PROMPT`，不用视频 prompt）→ ffmpeg 并轨 → Wan 回到 GPU。

Foley 钉了旧版 transformers，venv 与 Wan 分开。Sidecar 常驻，权重闲时放 CPU。`WAN22_FOLEY_REQUIRED=0`（默认）时 Foley 失败仍上传无声片；`=1` 则 `failed` / `foley_failed`。需要 `WAN22_OFFLOAD=none`。

## 已知约束

与旧服务相同：I2V 锁首帧、CFG=1 负向词无效、单卡串行。成片对象键为 `{WAN22_S3_PREFIX}{task_id}.mp4`。详见仓库里 `wan22-api/README.md` 的推理说明。
