# Wan 2.2 GPU Worker

无情的生成机器：从 SQS 拉任务 → 生成 → 上传 S3 → 写状态文件 → webhook 回调 → 删消息。部署在 vast.ai 上，接单走 `lambda_api`（Lambda + SQS），本机不暴露任何生成接口。

没有队列（进程内单执行位，忙时消息退回 SQS）、没有 Redis、没有 SQLite 留档。幂等靠 S3 状态文件，崩溃恢复靠 SQS 重投。推理是 WAMU Lightning I2V，attention 默认用 SageAttention 替换 SDPA（`WAN22_SAGE=1`，未安装自动回退 SDPA，每单 `attention` 日志记录实际调用/回退次数）。480p 画布约 480×832；720p 约 720×1248；`1080p` 先按 720p 生成，再用 compact（realesr-general-x4v3）超到约 1080×1872。

```
server/
  start.sh deploy.sh setup-aws.sh requirements.txt .env.example README.md JAVA.md
  wan22/
    config.py
    log.py        按日滚动的 wan22.log
    api/          /health /ready（仅本机探活），lifespan 里启动 SQS 消费
    tasks/        runner.py 单任务流水线 + sqs_consumer.py SQS 长轮询
    infer/        Wan pipeline
    media/        下图、上传、状态文件、webhook
    net/          URL 白名单 / SSRF
```

## 接单协议

Java 只跟 `lambda_api`（API Gateway + Lambda）打交道，协议见 [lambda_api/README.md](lambda_api/README.md) 与 [JAVA.md](JAVA.md)。SQS 消息体 = 入队 JSON：`task_id` + `prompt` / `negative_prompt` / `image_url` / `last_image_url` / `duration` / `resolution` / `webhook_url` / `seed` / `steps` / `quality` / `audio`。

本机 HTTP 只剩 `/health`（探活）与 `/ready`（模型未加载时 503），不接生成请求。

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

`deploy.sh` 装完依赖后会调用 `setup-aws.sh`：交互式问专用 IAM 用户的 Access Key / Secret / Region，先用 STS 验一次，通过后写进 `.env`（`chmod 600`，已在 `.gitignore`）。空回车跳过；非交互环境（CI / 云-init）只读环境变量、不落盘；`WAN22_AWS_SETUP=0 ./deploy.sh` 可整体跳过。key 轮换直接 `./setup-aws.sh --force`，不用重跑部署。

`start.sh` 监听 `0.0.0.0:8000`，长时间跑请用 `screen`。第一次默认跳过 Foley。1080p 需要 compact 超分：`./deploy.sh` 默认会装（`WAN22_UPSCALE_SKIP=0`）。

vast.ai 上如果 `nvcc --version` 是 13.2，而 `python -c 'import torch; print(torch.version.cuda)'` 是 12.8，SageAttention 扩展编译会因 CUDA 主版本不一致失败。PyTorch wheel 的 CUDA 运行库不等于系统 CUDA Toolkit；仅设置 `CUDA_HOME=/usr/local/cuda` 不能解决版本不一致。`deploy.sh` 会验证实际 nvcc 版本，优先使用匹配的 `CUDA_HOME` / `CUDA_PATH` 或 `/usr/local/cuda-12.8`；root 用户缺少 12.8 时会下载 NVIDIA runfile，仅补装 Toolkit，并恢复原有 `/usr/local/cuda` 软链。找到后，两套 SageAttention 扩展都使用该编译环境；找不到则跳过编译，保留 SDPA 回退。自动补装需要下载较大的安装包并预留磁盘空间。

更新 GPU 机器上的脚本后重跑 `./deploy.sh` 即可。自定义安装位置可用 `CUDA_HOME=/path/to/cuda-12.8 ./deploy.sh`；只想跳过 SageAttention 安装可用 `WAN22_SAGE_SKIP=1 ./deploy.sh`（也可写进 `.env`）。

96GB 显存优化与手动 GPU 对照测试见 [BENCHMARK.md](BENCHMARK.md)。当前代码不更改模型、步数、帧数、CFG、VAE 精度或默认超分插值。

- `WAN22_EXPERT_RESIDENCY=auto`：所有分辨率按实际生成画布和帧数决定驻留。总显存 >=90 GiB，且空闲显存满足 `WAN22_RESIDENT_HEADROOM_GIB × max(1, 像素数/898560 × 帧数/81)` GiB 时双 expert 常驻，预算系数默认 16。这是可调启发式，不是 OOM 保证。`on` 强制常驻；`off` 强制分时加载（也包含短片）。新变量优先；未设置时兼容旧 `WAN22_1080P_EXPERT_RESIDENCY`，但旧变量现在应用于所有分辨率。分时加载只在 expert 变化时执行，先卸载旧模型再加载新模型；Accelerate offload 模式不再混入手动搬运。
- `WAN22_SAGE_BACKEND=auto|sdpa|sage2|sage3`：auto 保持原 Sage3 → Sage2 → SDPA 顺序，`WAN22_SAGE=0` 仍强制关闭。每单记录实际后端、Sage/SDPA 调用数、回退次数。某个形状发生内核错误后只记录一次并熔断到 SDPA；CUDA OOM 原样抛出。显式后端安装失败仍允许服务回退，benchmark 则拒绝把回退结果当作该后端的成绩。
- `WAN22_PROFILE=auto|on|off`：auto 保留仅 1080p 详细计时，on 覆盖全部分辨率，off 关闭阶段同步计时。`generation profile` 记录 prompt、latent_prepare、denoise、vae_decode 和 allocated/reserved 峰值；所有请求都有 `generation` 和 `attention` 日志。S3 成功状态的 `timing.generation` 包含这些生成指标，`timing.transfer_s` 记录后处理 Wan 搬运。阶段统计不包含全部调度/清理开销，不要求相加等于总耗时。
- `WAN22_1080P_QUANT=baseline`：沿用 `WAN22_QUANT`，旧 auto 也使用基线。之前在 RTX PRO 6000 Blackwell 上的 FP8 试验出现非有限数值/黑屏，因此 `float8dq` 仅保留为显式实验，不能作为默认提速方案。独立 FP8 expert 从 BF16 加载、融合 LoRA、跳过 FP32 例外层后量化；未使用的 pair 留在 CPU，增加主机内存占用。
- `WAN22_1080P_UPSCALE_KEEP_WAN=auto`：无 Foley 的 1080p、总显存 >=90 GiB、清理缓存后空闲 >=16 GiB 时共享 GPU。超分 CUDA OOM 时终止 sidecar、卸载 Wan 并重试一次。所有路径都在 Foley/恢复 Wan 前让超分权重回到 CPU；失败则杀掉 sidecar。CPU 权重和进程保留，下单无需重复导入和从磁盘加载；CUDA context 仍占少量显存，预算查询使用实际空闲值。
- `WAN22_UPSCALE_RESIZE=cpu_lanczos`：默认保留现有 Lanczos 缩放，4×结果在设备上转成 uint8 后回传，减少 D2H 数据量。实验项 `gpu_bicubic` 在 GPU 上先缩至最终尺寸，再回传 uint8；这会改变插值，应比较皮肤、发丝、文字、运动闪烁后再采用。

改 `.env` 后重启 worker 即可，无需重新部署依赖。旧日志里的 445.676 / 552.732 / 777.020 秒仅为历史参考，不能与不同输入、后端或计时设置的结果直接比较。

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

本机测试生成：起一个本地 SQS 替身或直接用 `sqs_test/send_task.py` 往真队列发一条（`DRY_RUN=1` 时成片是占位 mp4，不真上传）。

## 日志

写在 `WAN22_LOG_DIR`（默认 `./logs`）：

- 当前：`logs/wan22.log`
- 按日滚动：`logs/wan22.log.YYYY-MM-DD`（本地时区午夜，默认保留 30 天）
- 同时打到 stdout，方便 journald / 终端

记录接单、拒单、推理开始/结束（含 seed、分辨率、耗时）、上传 URL、webhook 成败。uvicorn access 也进同一文件。

GPU 本机异常告警、Mac 上每小时/每日抽日志，见 [ops/README.md](ops/README.md)。

## 任务流（SQS 接单，vast.ai）

配了 `SQS_QUEUE_URL` 即启动后台线程长轮询。进程里只有一个执行位：抢到位就跑「下图 → I2V →（1080p 超分）→（Foley）→ 上传 S3 → 写终态 → webhook → 删消息」；抢不到（上一单还在跑）消息 `ChangeMessageVisibility` 退回（`SQS_BUSY_BACKOFF` 秒后再投），进程内不排队。**失败不重试**，直接落 `failed` + 短码并回调。

- 幂等：重复投递先查 S3 状态文件，已有终态就删消息。
- 可见性：生成可能十几分钟，看门线程每 `SQS_HEARTBEAT_SECONDS`（默认 120s）续一次 `SQS_VISIBILITY_TIMEOUT`（默认 600s），执行位释放后删除消息。
- 崩溃恢复：无本地留档，进程重启后上一单的事务全靠消息重投（visibility 过期自动重现）。重复生成同一任务理论上可能（崩溃在 webhook 之后、删消息之前），由 task_id 幂等挡住二次回调。
- 坏消息：校验失败或 `ApproximateReceiveCount >= SQS_MAX_RECEIVE`（默认 5）时写 `failed` 状态、发一次失败 webhook、删除消息，不会无限重投。队列上仍可叠加 DLQ。
- 凭证：vast.ai 没有 IAM Role，用 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`（与 S3 上传共用同一身份），IAM 需要 `sqs:ReceiveMessage` / `DeleteMessage` / `ChangeMessageVisibility`。`./deploy.sh` 会引导写入，也可单独跑 `./setup-aws.sh`。

## 任务状态文件

GPU 不看外部 API 也能上报进度：每个任务在 S3 写**一个**状态对象 `{WAN22_STATUS_PREFIX}/status/{task_id}.json`（默认前缀同 `WAN22_S3_PREFIX`，即 `video/wan22/status/...`）。

- 用单 key 覆盖写，不做 `.pending` → `.generating` 改名（S3 没有 rename，copy+delete 中间会查到 0 或 2 个文件）；`PutObject` 原子，永远只读到一个对象，还能塞下 `duration`/`resolution`/`seed`/`video_url`/`error`/`receive_count`/入参。
- 写入点：接单写 `pending` → 推理前写 `generating` → 终态写 `succeeded`/`failed`（在 webhook **之前**写，Java 回调里立刻查也能看到）。幂等判断也读它。
- 状态写入是旁路：S3 报错只记日志，绝不影响生成、上传和回调。`WAN22_S3_BUCKET` 没配则不写。
- 消费端：`lambda_api` 的 `GET /v1/tasks/{id}` 直接读这个对象（见 [lambda_api/README.md](lambda_api/README.md)），`generating` 超时判定为 `stalled`。



## 音频（可选）

`deploy.sh` 装好后，worker 在成片上传前：把 Wan 挪到 CPU → HunyuanVideo-Foley XL 看视频出 wav（固定 `WAN22_FOLEY_PROMPT`，不用视频 prompt）→ ffmpeg 并轨 → Wan 回到 GPU。

Foley 钉了旧版 transformers，venv 与 Wan 分开。Sidecar 常驻，权重闲时放 CPU。`WAN22_FOLEY_REQUIRED=0`（默认）时 Foley 失败仍上传无声片；`=1` 则 `failed` / `foley_failed`。需要 `WAN22_OFFLOAD=none`。

## 1080p

`resolution=1080p` 先按 720p 生成，再用 compact（`realesr-general-x4v3`）4×超分并缩至 1.5×，竖屏约 1080×1872，然后可选 Foley、上传。显存充足且无 Foley 时 Wan 留在 GPU，否则临时放 CPU。超分权重空闲时留在 CPU，进程复用。超分失败返回 `upscale_failed`，不会把 720p 片标为 1080p 成功。GPU 缩放的可选项与对照方式见上文及 [BENCHMARK.md](BENCHMARK.md)。

## 已知约束

与旧服务相同：I2V 锁首帧、CFG=1 负向词无效、单卡串行。成片对象键为 `{WAN22_S3_PREFIX}{task_id}.mp4`。详见仓库里 `wan22-api/README.md` 的推理说明。

启动与异常输出：启动预加载只加载权重，不创建视频任务，也不上传 S3。配置 SQS 后，consumer 会自动领取已有消息；`receive=2` 代表重新投递，并非预热请求。首次任务可等待预加载锁，因此 runner 的生成计时可能包含冷启动。至少 90 GiB 且实际空闲至少 80 GiB、无 offload 时，基线初始化先上 GPU 再融合 LoRA 和量化，避免在 CPU 上运行数分钟的矩阵运算；小卡/空闲不足沿用 CPU 路径，GPU 初始放置 OOM 则回退 CPU。VAE 精度和 tiling 不变。

生成帧在导出前检查：空帧、NaN/Inf 或整段全零黑屏会导致任务失败，不进入超分或上传，也不会标为 succeeded。单个正常黑帧不影响其它有效帧。该检查只拦截明显异常，不代表全面画质验收。
