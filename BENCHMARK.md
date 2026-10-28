# Wan22：96GB 显存验证

此轮只完成本地代码/CPU/FFmpeg 验证；GPU 速度、峰值和画质由用户在 RTX PRO 6000 Max-Q 上验证。没有远程运行测试或修改服务器配置。

Git 基线为 `codex/wan22-baseline`（`7a44dd2`），实现分支为 `codex/wan22-96gb`。多数对照用同一实现分支的开关即可，避免代码版本成为额外变量。基线分支用于完整回退；新 benchmark 不存在于旧基线。

## 本轮范围

| 改动 | 默认行为 | 待 GPU 验证 |
| --- | --- | --- |
| 所有分辨率按实际负载/空闲显存选择双 expert 常驻 | auto，预算系数可调 | 720p/15s 和近方图峰值 |
| 分时加载只在 expert 切换时发生，先卸载再加载 | 开启 | 搬运时间、长短任务连续运行 |
| Sage 后端可选、调用统计、失败形状熔断 | auto 保持原选择顺序 | 加速实际命中、不同后端画质 |
| 超分进程复用，闲时权重回 CPU | 开启，释放失败则终止进程 | 连续任务及 Foley 混合负载 |
| 超分在设备上量化成 uint8 再回传 | 开启，保留 CPU Lanczos | 与原版成片比较 |
| 超分先在 GPU 缩至最终尺寸 | **关闭**；`gpu_bicubic` 实验开关 | 画质/时序变化、超分耗时 |
| `torch.compile` | **生产代码未开启**；仅 benchmark 可选 | 编译成本、预热后速度、兼容性 |
| BF16 expert | **默认仍 INT8**；benchmark `--quant none` | 显存与速度 |

尚未改变 MP4 中间传递、VAE tiling/精度、采样步数/帧数、模型/LoRA、Foley 模型和任务并发。原始帧直传与任务流水线重构应另做实验，避免和本轮混在一起。

## 运行方式

在 GPU 机的 `server` 部署目录中执行。先让生产 worker 停止接单并等待当前任务完成，避免 benchmark 与服务同时占卡；不要为了跑测试重新执行会更新依赖的 deploy.sh。

```bash
# 使用与 worker 相同的路径/依赖；benchmark 不自动读取 .env。
set -a
source .env
set +a
source ./worker-env.sh
source "$WAN22_VENV_DIR/bin/activate"

# 换成自己的同一张图片和 UTF-8 提示词文件。先 5s 排兼容性，再测 15s。
COMMON=(--image /workspace/test1.jpg --prompt-file /workspace/prompt.txt
        --duration 5 --steps 6 --seed 42 --resolution 720p
        --warmup 1 --runs 1 --attention auto --quant int8wo)

# A/B：仅切换驻留策略。off 是本分支已修复的分时策略，不是旧版逐步调度实现。
python benchmark.py "${COMMON[@]}" --residency off --output data/bench/staged-5s
python benchmark.py "${COMMON[@]}" --residency on  --output data/bench/resident-5s
```

输出目录必须不存在，防止覆盖已有结果。每个命令是一个新进程，默认包含至少一次完整预热；不会启动 API/SQS、上传 S3 或发送 webhook，也不生成 Foley。`--resolution 1080p` 会执行真实超分，要求该侧环境/模型已安装。

每个目录保留：

- `report.json`：输入哈希、Git commit/dirty 状态、依赖版本/可用的源码 commit、参数、模型路径、加载时间、每单耗时、中位数、实际 attention 调用和帧数/尺寸。
- `run-XX.mp4` 与首/中/尾帧 PNG，用于并排检查；解码检查和 PNG 导出在计时区间外。
- `logs/wan22.log`：阶段日志和错误。
- report 中的 nvidia-smi 每秒采样：所有可见物理 GPU 的 UUID、显存、功耗/功耗限制、利用率、温度、时钟。PyTorch peak 只统计 Wan 进程，超分进程要参考 nvidia-smi；秒级采样可能漏掉瞬时峰值。

每次对照固定图片、prompt、seed、时长、steps、FPS、CFG、LoRA、权重及依赖版本。`--profile off` 是默认吞吐测试；定位瓶颈时另跑一次 `--profile on`，不要混用两种计时模式比较速度。`load_s` 不计入单次生成；编译是 lazy 的，第一次预热含编译成本。预热的超分进程结束后会保持 CPU 权重，后续测量复用它。

## 推荐实验顺序

1. **驻留**：上面的 A/B。5s 通过后把 COMMON 的时长换成 15，再比较 `off / on / auto`。auto 预算为启发式；看实际 `generation.resident`，不要把 auto 理解为保证开启。长片和近方形画布要分别验证。
2. **Attention**：相同驻留、INT8 下比较 `--attention sdpa / sage2 / sage3`。auto 保留历史后端选择；明确指定某个后端却加载失败、或没有成功 Sage 调用时 benchmark 会失败，不会把 SDPA 回退冒充该后端成绩。检查 `fallback_calls`；部分回退的结果也应标明。
3. **超分**：以相同参数、seed、后端分别跑 `--resolution 1080p --resize cpu_lanczos` 和 `--resolution 1080p --resize gpu_bicubic`，使用不同输出目录。比 `postprocess_s` 和成片运动细节，不能只比较静态锐度。
4. **量化**：相同后端/驻留下比较 `--quant int8wo / none`，后者只取消 expert 量化，文本编码器仍沿用配置；主机 RAM 和显存都要观察。不要自动启用曾出现黑屏的 FP8。
5. **编译**：先完成 SDPA eager 基线，再加 `--compile default --attention sdpa --profile off --residency on`，仅接受 480p/720p，避免超分 OOM 回退把已编译模型搬到 CPU。此组合避开 Python attention 统计和动态搬运干扰；不保证 torch/torchao 组合都能编译。`max-autotune` 首次耗时可能很长，按需使用。必须和同样 SDPA 的 eager 比较，不能把换 attention 的收益算到编译上。

需要更稳定统计时 `--runs 3` 或更多，只比较预热后中位数，同时检查所有单次值。当前目录未包含任何 GPU 成绩，不预设提速百分比。

## 首个 block 算子诊断

完整 15 秒诊断过慢时，使用 `--block-profile --residency on`，仅执行首个实际专家 block 后停止，不生成视频。此模式不做预热/重复，要求 `--profile off --compile off` 和 480p/720p。完整命令、报告字段及 CausVid / LightX2V 方案评估见 [ACCELERATION.md](ACCELERATION.md)。

## 验收与回退

- 检查总耗时，以及不同阶段是否把耗时转移到了搬运/后处理；生成报告只覆盖生成和可选超分，不含下载、Foley、上传和队列耗时。
- 固定 seed 不代表不同精度/attention 的输出逐像素一致。看首尾图约束、脸部、手部、发丝、纹理闪烁及动作连续性；无 NaN/Inf/全黑只是最低门槛。
- 混合任务验证：长 720p → 短 480p → 1080p → 带 Foley 任务，观察显存是否回落。benchmark 不跑 Foley，该项需通过现有正常任务入口单独验收。
- 保留原 `.env`；新驻留设置优先于旧 `WAN22_1080P_EXPERT_RESIDENCY`，移除新变量后才会使用旧变量值。`WAN22_EXPERT_RESIDENCY=off` 强制分时加载，`WAN22_SAGE=0` 强制 SDPA，`WAN22_UPSCALE_RESIZE=cpu_lanczos` 关闭实验缩放；这些值不是原版逐位行为的保证。
- 要完整回到旧实现，在干净工作区切回 `codex/wan22-baseline` 并恢复原环境配置，重启 worker。不要重新下载权重或升级依赖。基准验收后再固定这组真实 GPU 环境的依赖版本；本轮不凭本地 CPU 环境猜测生产版本锁。

本地回归：

```bash
WAN22_DRY_RUN=1 WAN22_PRELOAD=0 WAN22_LOG_CONSOLE=0 python -m unittest discover -s tests
```
