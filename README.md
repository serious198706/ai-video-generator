# Wan 2.2 Adapter API

给 Java 后端调用的图生视频服务，协议对齐 a2e / Pixverse adapter。

Java 直接打 **GPU FastAPI**（`./start.sh`，默认 `:8000`）。流程只有一条直线：**POST 进来即开始生成 → 成片上传 S3/CloudFront → webhook 回调 Java**。没有队列、没有 Redis，单卡一次只跑一个任务，正在跑时再来的 POST 直接 `429`（由 Java 端决定何时重投）。任务过程落在本机 SQLite 里，只为留档，事后可查。推理仍是 WAMU Lightning I2V。480p 画布约 480×832；720p 约 720×1248；`1080p` 先按 720p 生成，再用 compact（realesr-general-x4v3）超到约 1080×1872。

```
server/
  start.sh deploy.sh requirements.txt .env.example README.md JAVA.md
  wan22/
    config.py
    log.py        按日滚动的 wan22.log
    api/          POST /v1/generate、查询任务、/health /ready
    tasks/        store.py 任务落库（SQLite 留档）+ runner.py 单任务流水线
    infer/        Wan pipeline
    media/        下图、上传、webhook
    net/          URL 白名单 / SSRF
```

## 协议

Java 调用端见 [JAVA.md](JAVA.md)（提交 / 查询 / webhook / 探活）。Base URL 用 GPU 的 `http://<host>:8000`。

`POST /v1/generate`，`application/json`，无鉴权。校验 HTTPS/白名单后立刻 `202`，同时后台线程已经在下图并推理（不要同步死等成片）：

```json
{ "id": "…", "task_id": "…", "status": "running" }
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

`GET /v1/tasks/{id}` 读 SQLite 里的任务（接口没变，重启后旧任务照样查得到）。`/health` 探活；`/ready` 在模型未加载时 503。`/docs` 默认关闭。

图片 URL 必须 https 且解析到公网（防 SSRF）。`WAN22_IMAGE_HOSTS` 有值才限制图床域名。Webhook 走 `WAN22_WEBHOOK_HOSTS`，**允许内网 IP**（Java 就在内网），但仍拒绝链路本地 / `169.254.169.254`；该项为空则不限制 host。

Webhook body 与任务查询字段一致：`id`、`task_id`、`status`、`video_url`、`error`、`seed`、`duration`、`resolution`。成功和失败都回调一次，失败不重试推理。失败时 `error` 仅为短码：`generate_failed` / `upscale_failed` / `foley_failed` / `upload_failed` / `download_failed` / `interrupted`。配置了 `WAN22_WEBHOOK_SECRET` 时带 `X-Wan-Signature: sha256=…`。

## 配置

复制 `.env.example` 为 `.env`（不入库）。GPU `./start.sh` 会 source 它。

必填：

- `WAN22_S3_BUCKET`（非 DRY_RUN 启动时必填）

`WAN22_IMAGE_HOSTS` / `WAN22_WEBHOOK_HOSTS` 可选。逗号分隔，`.cloudfront.net` 这种写法按后缀匹配。**空或不设则不限制 host**；仍要求 https。图片继续拒绝私网（防 SSRF）；webhook 允许内网，拒绝链路本地。测试时把 `WAN22_IMAGE_HOSTS` 留空或改成你的图床域名。

成片：`WAN22_S3_BUCKET` / `WAN22_S3_REGION` / `WAN22_S3_PREFIX`，回传 URL 用 `WAN22_S3_PUBLIC_BASE_URL`（CloudFront）。凭证走 GPU 机 IAM Role，或标准的 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`。

## 启动

路径全部落在本目录（`server/`）：venv、权重、日志、缓存都不写死 AutoDL / EC2 / vast.ai。Python 从 PATH 找 `python3` 再找 `python`；镜像里已经有 CUDA torch 时 `deploy.sh` 会复用，不再强制装 cu128。

```bash
cp .env.example .env
# 改 hosts / S3。国内机器再打开 .env 里的镜像项。
./deploy.sh
./start.sh
```

`start.sh` 监听 `0.0.0.0:8000`，长时间跑请用 `screen`。第一次默认跳过 Foley。1080p 需要 compact 超分：`./deploy.sh` 默认会装（`WAN22_UPSCALE_SKIP=0`）。

干净 Ubuntu 26.04（裸金属，没有 DLAMI）先做系统层，再走上面的 `deploy.sh`：

```bash
# 可选：在现网 EC2 上跑，把驱动 / Python / torch 版本打出来对照
./inspect-host.sh

# 裸金属：root 安装 python/git/ffmpeg；驱动未装时先 --install-driver 并 reboot
sudo ./bootstrap-ubuntu.sh --install-driver --skip-deploy
# reboot 后
sudo ./bootstrap-ubuntu.sh
```

`deploy.sh` 会在本目录建 Wan venv 并拉 WAMU 权重。探测到现成 CUDA torch 就 `--system-site-packages` 复用；没有再 pip 装 cu128。Foley 默认跳过（`WAN22_FOLEY_SKIP=1`）；`start.sh` 看到 Foley python 才打开配音。不要 Foley：`.env` 里 `WAN22_FOLEY_ENABLE=0`。compact 超分默认安装；不要 1080p：`WAN22_UPSCALE_SKIP=1` 且 `WAN22_UPSCALE_ENABLE=0`。

本机 dry-run（不需要 GPU）：

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

记录接单、拒单（400/429/503）、推理开始/结束（含 seed、分辨率、耗时）、上传 URL、webhook 成败、进程重启后判死的任务。uvicorn access 也进同一文件。

GPU 本机异常告警、Mac 上每小时/每日抽日志，见 [ops/README.md](ops/README.md)。

## 任务

无队列。进程里只有一个执行位：`POST /v1/generate` 抢到位就建档（状态直接是 `running`）并在后台线程跑「下图 → I2V →（1080p 超分）→（Foley）→ 上传 S3 → webhook」；抢不到就 `429`，要不要重投由 Java 端决定。**失败不重试**，直接落 `failed` + 短码并回调。

任务落在 SQLite（`WAN22_TASK_DB`，默认 `<WAN22_DATA_DIR>/tasks.db`，WAL 模式）里，只做留档：`/v1/tasks/{id}` 从库里读，进程重启后历史任务仍可查。重启时库里残留的 `running` 会被一次性判死为 `failed` / `interrupted` 并补发 webhook，免得 Java 端一直等。想事后翻账直接查库：

```bash
sqlite3 data/tasks.db "select created_at,status,error,resolution,seed,video_url from tasks order by created_at desc limit 20"
```

## 音频（可选）

`deploy.sh` 装好后，worker 在成片上传前：把 Wan 挪到 CPU → HunyuanVideo-Foley XL 看视频出 wav（固定 `WAN22_FOLEY_PROMPT`，不用视频 prompt）→ ffmpeg 并轨 → Wan 回到 GPU。

Foley 钉了旧版 transformers，venv 与 Wan 分开。Sidecar 常驻，权重闲时放 CPU。`WAN22_FOLEY_REQUIRED=0`（默认）时 Foley 失败仍上传无声片；`=1` 则 `failed` / `foley_failed`。需要 `WAN22_OFFLOAD=none`。

## 1080p

`resolution=1080p` 不会把 Lightning 画布拉到 1080×1872。worker 先按 720p 出无声片，把 Wan 挪到 CPU，compact sidecar（`realesr-general-x4v3`，4× 后再 Lanczos 到 1.5×，竖屏约 1080×1872）超分，再配 Foley、上传。超分失败短码 `upscale_failed`，不退回 720p。没装超分时 POST 1080p 返回 503。第一次 1080p 会加载超分权重，后面复用。

## 已知约束

与旧服务相同：I2V 锁首帧、CFG=1 负向词无效、单卡串行。成片对象键为 `{WAN22_S3_PREFIX}{task_id}.mp4`。详见仓库里 `wan22-api/README.md` 的推理说明。
