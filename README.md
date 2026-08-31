# Wan 2.2 Adapter API

给 Java 后端调用的内网图生视频服务，协议对齐 a2e / Pixverse adapter。

本目录与旧的 `wan22-api/`（multipart + SQLite）隔离。推理仍是 WAMU Lightning I2V，画布约 480×832，成片上传到 S3（可用 CloudFront 域名回传 `video_url`）。

```
server/
  start.sh deploy.sh requirements.txt .env.example README.md
  wan22/
    config.py
    log.py        按日滚动的 wan22.log
    api/          FastAPI 路由与请求体
    queue/        SQLite 任务与 worker
    infer/        Wan pipeline
    media/        下图、上传、webhook
    net/          URL 白名单 / SSRF
```

## 协议

Java 调用端见 [JAVA.md](JAVA.md)（提交 / 查询 / webhook / 探活）。

`POST /v1/generate`，`application/json`，无鉴权。立刻 `202`：

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
  "lastImage": null
}
```

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `image` | 是 | HTTPS 图片 URL（S3 / CloudFront），服务端下载 |
| `prompt` | 否 | 空则用 `WAN22_DEFAULT_PROMPT` |
| `negativePrompt` | 否 | 会传给 pipeline；`guidance_scale=1` 时不生效 |
| `duration` | 否 | `(0, 15]` 秒，默认 5 |
| `resolution` | 否 | `540p` / `720p` / `1080p`，本轮只记录，画布仍 480×832 |
| `webhookUrl` | 否 | 成功和失败都会 POST |
| `steps` / `quality` / `seed` / `lastImage` | 否 | Wan 私有字段；`quality` 为导出 1–10 |

`GET /v1/tasks/{id}` 可轮询。`/health` 恒 200；`/ready` 模型未就绪时 503。`/docs` 默认关闭。

图片 URL 必须解析到公网（防 SSRF）。Webhook 走独立白名单，**允许内网 IP**（Java 就在内网），但仍拒绝链路本地 / `169.254.169.254`。

Webhook body 与任务查询字段一致：`id`、`task_id`、`status`、`video_url`、`error`、`seed`、`duration`、`resolution`。失败时 `error` 仅为短码：`generate_failed` / `foley_failed` / `upload_failed` / `download_failed` / `interrupted`。配置了 `WAN22_WEBHOOK_SECRET` 时带 `X-Wan-Signature: sha256=…`。

## 配置

复制 `.env.example` 为 `.env`（不入库）。`./start.sh` 会 source 它。

必填白名单：

- `WAN22_IMAGE_HOSTS`（空则拒绝下图）
- `WAN22_WEBHOOK_HOSTS`（请求带了 `webhookUrl` 但此项为空则 400）
- `WAN22_S3_BUCKET`（非 DRY_RUN 启动时必填）

列表用逗号分隔，`.cloudfront.net` 这种写法按后缀匹配。

成片：`WAN22_S3_BUCKET` / `WAN22_S3_REGION` / `WAN22_S3_PREFIX`，回传 URL 用 `WAN22_S3_PUBLIC_BASE_URL`（CloudFront）。凭证走 GPU 机 IAM Role，或标准的 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`。

## 启动

需要本机 SQLite（默认 `{WAN22_DATA_DIR}/queue.sqlite`，不另起中间件）。GPU 机生产：

```bash
cp .env.example .env
# 改 hosts / S3
./deploy.sh
./start.sh
```

`deploy.sh` 会装 Wan venv、WAMU 权重，以及独立的 Foley venv（`/opt/foley-venv`）和 HunyuanVideo-Foley XL 权重。Foley 沿用已有解释器，不重建 venv；3.14 上会改用带轮子的 Pillow / NumPy，避免源码编译。只要 Foley 的 python 在，`start.sh` 默认打开配音。不要 Foley：`.env` 里 `WAN22_FOLEY_ENABLE=0`，或 `WAN22_FOLEY_SKIP=1 ./deploy.sh`。

无 GPU 联调：

```bash
export WAN22_DRY_RUN=1
export WAN22_IMAGE_HOSTS=127.0.0.1,localhost
export WAN22_WEBHOOK_HOSTS=127.0.0.1,localhost
uvicorn wan22.api.app:app --port 8000
```

`DRY_RUN` 写占位 mp4，不上传 S3，`video_url` 为 `http://127.0.0.1/dry-run/{id}.mp4`。

## 日志

写在 `WAN22_LOG_DIR`（默认 `./logs`）：

- 当前：`logs/wan22.log`
- 按日滚动：`logs/wan22.log.YYYY-MM-DD`（本地时区午夜，默认保留 30 天）
- 同时打到 stdout，方便 journald / 终端

记录入队、拒单（400/429/503）、推理开始/结束（含 seed、分辨率、耗时）、上传 URL、webhook 成败、进程重启中断的任务。uvicorn access 也进同一文件。

## 队列

本机 SQLite `{WAN22_DATA_DIR}/queue.sqlite`（可用 `WAN22_QUEUE_DB` 改路径）。Worker 轮询出队。深度（排队 + 正在跑）≥ `WAN22_QUEUE_MAX`（默认 50）返回 429。进程重启时若有 `running`，该任务标 `failed` / `interrupted` 并 webhook；已入队的继续消费。

## 音频（可选）

`deploy.sh` 装好后，worker 在成片上传前：把 Wan 挪到 CPU → HunyuanVideo-Foley XL 看视频出 wav（固定 `WAN22_FOLEY_PROMPT`，不用视频 prompt）→ ffmpeg 并轨 → Wan 回到 GPU。

Foley 钉了旧版 transformers，venv 与 Wan 分开。Sidecar 常驻，权重闲时放 CPU。`WAN22_FOLEY_REQUIRED=0`（默认）时 Foley 失败仍上传无声片；`=1` 则 `failed` / `foley_failed`。需要 `WAN22_OFFLOAD=none`。

## 已知约束

与旧服务相同：I2V 锁首帧、CFG=1 负向词无效、单卡串行。成片对象键为 `{WAN22_S3_PREFIX}{task_id}.mp4`。详见仓库里 `wan22-api/README.md` 的推理说明。
