# Wan2.2 加速判断与下一次诊断

2026-10-28。硬件 RTX PRO 6000 Blackwell Max-Q 96GB；本次只做本地代码和 CPU 测试，没有连接或测试远程 GPU。

## 当前证据

用户提供的 `profile-15s/report.json`：总计 749.27 秒；去噪 613.03 秒（高噪声 306.20、低噪声 306.83）；latent 准备 46.34 秒；VAE 解码 77.33 秒；导出 6.92 秒。加载 47.18 秒另计。已使用 4 步、CFG 1、Sage2、int8 weight-only 和双专家常驻。

这说明主要瓶颈仍在去噪，但不足以判断是 attention、矩阵乘还是量化转换。不能依据 int8 标签直接断定它更慢，也不能把 Sage/SDPA 调用次数当成耗时占比。

按官方 A14B 的时间压缩 4、空间压缩 8 和 patch (1,2,2) 计算，实际输入 832×1248、237 帧对应 60×52×78 = 243,360 个 token；77 帧则是 81,120 个。token 数增加 3 倍，dense self-attention 的二次项计算量约增至 9 倍。这是配置推算，不是算子实测，也不表示总耗时一定增加 9 倍。新诊断记录真实 block 输入形状以核对。

来源：[官方 A14B 配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_i2v_A14B.py)、[官方 attention](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/attention.py)。官方原版的采样设置不应覆盖当前 Lightning 权重的设置。

## 只跑一个真实 block

在原有服务器环境加载 `worker-env.sh` 后，从 server 目录执行；输入沿用之前的对照文件，输出目录必须不存在：

```bash
python benchmark.py \
  --image /path/to/input.png \
  --prompt-file /path/to/prompt.txt \
  --output /path/to/block-15s \
  --resolution 720p --duration 15 --steps 4 --seed 42 \
  --residency on --attention sage2 --quant int8wo \
  --block-profile --warmup 0 --runs 1
```

`--block-profile` 只执行实际被调用专家的第一个 block，然后结束；不执行其余 block、后续去噪步、VAE 解码、视频导出或超分。此模式忽略 warmup/runs，始终一次冷诊断。仍需加载模型、编码提示词、准备 latent 和执行到首个 block 之前的操作，不能承诺固定完成时间。

结果：

- `report.json`：环境、配置、加载/诊断时间、attention 计数、GPU telemetry；measurement_kind 为 first_block_diagnostic。
- `block-profile.json`：输入形状/dtype、权重张量类型、算子 CPU/GPU 时间，以及 `wan_block/attn1`、`attn2`、`ffn` 等嵌套模块范围（名称跟随实际模型）。
- `block-trace.json`：PyTorch profiler Chrome trace，可本地导入 Perfetto 等查看时间线。

真实异常会失败退出；未执行 block 或未采集到 CUDA events 也会失败，不会冒充成功。若没有 CUDA events，检查运行环境的 CUPTI/profiler 支持。文件不记录输入 tensor 内容或提示词原文，但包含 tensor 形状和模型/输入文件路径。

这是冷 block 的定位工具，有 profiler/JIT 开销，且只代表首个专家的首个 block；不能当成全片速度或画质测试。嵌套 total 时间重叠，不能相加。优先查看 attention kernel 与 linear/matmul/量化转换的 self 时间，再结合模块范围和 trace 判断。后续任何加速改动仍须用完整视频对照速度、画质与稳定性。

## 外部方案取舍

### CausVid：不直接接入当前生产流程

[CausVid](https://github.com/tianweiy/CausVid) 将双向扩散蒸馏成因果自回归模型，需要对应架构、训练与权重，不是通用推理开关。公开示例基于 Wan2.1 T2V 1.3B，I2V 仍列为 TODO。论文 50→4 步的收益不能再套在当前已经 4 步的 A14B 上。

仓库 [LICENSE.md](https://github.com/tianweiy/CausVid/blob/master/LICENSE.md) 为 CC BY-NC-SA 4.0，商业集成需另行解决授权。这条路线适合作为换模型研究，不作为当前 WAMU 的即插即用优化。

### LightX2V / LightWan2.2-A14B：更值得做独立候选验证

[Wan 官方 README](https://github.com/Wan-Video/Wan2.2) 列出了 [LightX2V](https://github.com/ModelTC/LightX2V)。目前其 [LightWan2.2-A14B 模型卡](https://huggingface.co/lightx2v/LightWan2.2-A14B) 提供面向 Blackwell 的 NVFP4 量化感知蒸馏与稀疏注意力方案，含 I2V 和 4 步专家调度。模型卡标注 Apache-2.0。

它比 CausVid 更贴近当前任务，但仍要更换专用权重、安装对应 kernel 和运行环境。当前 WAMU 权重/LoRA 的效果、首尾帧能力、15 秒长度和 6000 Max-Q 支持情况都需要实际核对。不能默认把现有 LoRA 叠加到新量化权重就等价。

官方表格是 RTX 5090 上对原版 40 NFE 的比较；表格未完整列出本项目需要的时长/画幅条件。不能拿公布的 720p 延迟或 100× 直接预测我们的 15 秒任务。当前 CUDA 12.8 环境也不能默认兼容其推荐的 CUDA 13.0 镜像，应使用独立环境。

建议顺序：先收集一个 block，确定当前瓶颈；若 attention 占主导，优先独立验证上述稀疏方案；若 linear/转换占主导，先比较矩阵乘后端与量化实现。LightWan 的验证必须单列“更换模型”与画质结果，不能混入现有模型的纯性能对照。

官方 TI2V-5B 也可作为另一个换模型方案，但 VAE/架构不同，无法保持现有 A14B 权重与 LoRA。本次没有替换任何生产权重，也未声称已取得新的 GPU 加速结果。
