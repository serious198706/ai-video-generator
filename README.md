# Wan 2.2 Adapter API

给 Java 后端调用的图生视频服务，协议对齐 a2e / Pixverse adapter。

Java 直接打 **GPU FastAPI**（`./start.sh`，默认 `:8000`）。`POST /v1/generate` 在本机收下任务并推理，任务记在进程内存里，不再用 Redis / ElastiCache。推理仍是 WAMU Lightning I2V。480p 画布约 480×832；720p 约 720×1248；`1080p` 先按 720p 生成，再用 SeedVR2 超到约 1080×1872，成片上传到 S3（可用 CloudFront 域名回传 `video_url`）。

```
server/
  start.sh deploy.sh requirements.txt .env.example README.md JAVA.md
  wan22/
    config.py
    log.py        按日滚动的 wan22.log
    api/          POST /v1/generate、查询任务、/health /ready
    queue/        本机内存任务 + 串行 worker
    infer/        Wan pipeline
    media/        下图、上传、webhook
    net/          URL 白名单 / SSRF
```

## 协议

Java 调用端见 [JAVA.md](JAVA.md)（提交 / 查询 / webhook / 探活）。Base URL 用 GPU 的 `http://<host>:8000`。

`POST /v1/generate`，`application/json`，无鉴权。立刻 `202`（校验 HTTPS/白名单后入本机任务表，后台下图并推理）：

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
| `image` | 是 | HTTPS 图片 URL（JPEG / PNG / WebP / GIF 等）；接口校验后由本机 worker 下载并转成 JPEG |
| `prompt` | 否 | 空则用 `WAN22_DEFAULT_PROMPT` |
| `negativePrompt` | 否 | 会传给 pipeline；`guidance_scale=1` 时不生效 |
| `duration` | 否 | `(0, 15]` 秒，默认 5 |
| `resolution` | 否 | `480p` / `540p` / `720p` / `1080p`。480p 画布约 480×832；720p 约 720×1248；1080p 先 720p 再超分 |
| `webhookUrl` | 否 | 成功和失败都会 POST |
| `steps` / `quality` / `seed` / `lastImage` | 否 | Wan 私有字段；`quality` 为导出 1–10 |
| `audio` | 否 | 是否配 Foley，默认 `false`。不传或 `null` 都不配音；传 `true` 才跑 Foley |

`GET /v1/tasks/{id}` 读本机任务。`/health` 探活；`/ready` 在模型未加载时 503。`/docs` 默认关闭。

图片 URL 必须 https 且解析到公网（防 SSRF）。`WAN22_IMAGE_HOSTS` 有值才限制图床域名。Webhook 走 `WAN22_WEBHOOK_HOSTS`，**允许内网 IP**（Java 就在内网），但仍拒绝链路本地 / `169.254.169.254`；该项为空则不限制 host。

Webhook body 与任务查询字段一致：`id`、`task_id`、`status`、`video_url`、`error`、`seed`、`duration`、`resolution`。失败时 `error` 仅为短码：`generate_failed` / `upscale_failed` / `foley_failed` / `upload_failed` / `download_failed` / `interrupted`。配置了 `WAN22_WEBHOOK_SECRET` 时带 `X-Wan-Signature: sha256=…`。

## 配置

复制 `.env.example` 为 `.env`（不入库）。GPU `./start.sh` 会 source 它。

必填：

- `WAN22_S3_BUCKET`（非 DRY_RUN 启动时必填）

`WAN22_IMAGE_HOSTS` / `WAN22_WEBHOOK_HOSTS` 可选。逗号分隔，`.cloudfront.net` 这种写法按后缀匹配。**空或不设则不限制 host**；仍要求 https。图片继续拒绝私网（防 SSRF）；webhook 允许内网，拒绝链路本地。测试时把 `WAN22_IMAGE_HOSTS` 留空或改成你的图床域名。

成片：`WAN22_S3_BUCKET` / `WAN22_S3_REGION` / `WAN22_S3_PREFIX`，回传 URL 用 `WAN22_S3_PUBLIC_BASE_URL`（CloudFront）。凭证走 GPU 机 IAM Role，或标准的 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`。

## 启动

路径由 `worker-env.sh` 按机器决定：有 `/root/autodl-tmp` 走 AutoDL 数据盘和镜像 conda Torch；否则现网 EC2 的 `/opt` + `/data`。

```bash
cp .env.example .env
# 改 hosts / S3。AutoDL 不要改路径，不要设 WAN22_INSTALL_TORCH=1
./deploy.sh
./start.sh
```

AutoDL 权重、venv、日志在 `/root/autodl-tmp/wan22`。系统盘只有 30G。`start.sh` 监听 `0.0.0.0:8000`，长时间跑请用 `screen`。第一次默认跳过 Foley。1080p 需要 SeedVR2：`./deploy.sh` 默认会装（`WAN22_UPSCALE_SKIP=0`）。

干净 Ubuntu 26.04（裸金属，没有 DLAMI）先做系统层，再走上面的 `deploy.sh`：

```bash
# 可选：在现网 EC2 上跑，把驱动 / Python / torch 版本打出来对照
./inspect-host.sh

# 裸金属：root 安装 python/git/ffmpeg；驱动未装时先 --install-driver 并 reboot
sudo ./bootstrap-ubuntu.sh --install-driver --skip-deploy
# reboot 后
sudo ./bootstrap-ubuntu.sh
```

`deploy.sh` 会装 Wan venv 和 WAMU 权重。AutoDL 继承镜像 `torch 2.12.1+cu130`，不装 cu128；EC2 仍装 cu128。Foley 在 AutoDL 默认跳过（`WAN22_FOLEY_SKIP=1`）；现网检测到 Foley python 后 `start.sh` 默认打开。不要 Foley：`.env` 里 `WAN22_FOLEY_ENABLE=0`。SeedVR2 默认安装；不要 1080p：`WAN22_UPSCALE_SKIP=1` 且 `WAN22_UPSCALE_ENABLE=0`。

本机 dry-run（不需要 GPU / Redis）：

```bash
export WAN22_DRY_RUN=1
export WAN22_IMAGE_HOSTS=
uvicorn wan22.api.app:app --port 8000
```

`DRY_RUN` 写占位 mp4，不上传 S3，`video_url` 为 `http://127.0.0.1/dry-run/{id}.mp4`。

测试生成：

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"image":"https://dxxx.cloudfront.net/a.jpg","prompt":"she turns her head","duration":5}'
# 用返回的 id 轮询
curl -sS http://127.0.0.1:8000/v1/tasks/<id>
```

## 日志

写在 `WAN22_LOG_DIR`（默认 `./logs`）：

- 当前：`logs/wan22.log`
- 按日滚动：`logs/wan22.log.YYYY-MM-DD`（本地时区午夜，默认保留 30 天）
- 同时打到 stdout，方便 journald / 终端

记录接单、拒单（400/429）、推理开始/结束（含 seed、分辨率、耗时）、上传 URL、webhook 成败、进程重启后重试。uvicorn access 也进同一文件。

GPU 本机异常告警、Mac 上每小时/每日抽日志，见 [ops/README.md](ops/README.md)。

## 任务

进程内内存：待跑列表 + 任务字典。单卡串行。待跑数量 ≥ `WAN22_QUEUE_MAX`（默认 **500**）返回 429。失败最多重试 3 次（含首次）；`attempts >= 3` 则 `failed` + webhook。进程退出后内存任务清空。

## 音频（可选）

`deploy.sh` 装好后，worker 在成片上传前：把 Wan 挪到 CPU → HunyuanVideo-Foley XL 看视频出 wav（固定 `WAN22_FOLEY_PROMPT`，不用视频 prompt）→ ffmpeg 并轨 → Wan 回到 GPU。

Foley 钉了旧版 transformers，venv 与 Wan 分开。Sidecar 常驻，权重闲时放 CPU。`WAN22_FOLEY_REQUIRED=0`（默认）时 Foley 失败仍上传无声片；`=1` 则 `failed` / `foley_failed`。需要 `WAN22_OFFLOAD=none`。

## 1080p

`resolution=1080p` 不会把 Lightning 画布拉到 1080×1872。worker 先按 720p 出无声片，把 Wan 挪到 CPU，SeedVR2-3B FP8 sidecar 按短边 1.5× 超分（竖屏约 1080×1872），再配 Foley、上传。超分失败短码 `upscale_failed`，不退回 720p。没装超分时 POST 1080p 返回 503。第一次 1080p 会加载超分权重，后面复用。

## 已知约束

与旧服务相同：I2V 锁首帧、CFG=1 负向词无效、单卡串行。成片对象键为 `{WAN22_S3_PREFIX}{task_id}.mp4`。详见仓库里 `wan22-api/README.md` 的推理说明。
