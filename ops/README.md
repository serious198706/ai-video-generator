# Wan22 监控

三层：**GPU watchdog**（异常才发飞书）、**Mac 每小时拉日志**（有 ERROR/失败才发）、**Mac 每天 8:00 日报**（必发）。数量和平均时长由脚本从 `timing` 行计算，不靠 LLM 数。

Hermes 只负责到点跑脚本。飞书由脚本自己 POST（`--notify`）。

## 1. GPU：watchdog

同步仓库后（默认代码在 `/opt/server`）：

```bash
# /opt/server/.env
WAN22_FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/xxxx
# WAN22_FEISHU_SECRET=   # 群机器人开了签名才填
WAN22_SYSTEMD_UNIT=wan22-gpu

sudo chmod +x /opt/server/ops/watchdog.sh
# 若代码不在 /opt/server，改 unit 里的路径
sudo cp /opt/server/ops/wan22-watchdog.service /opt/server/ops/wan22-watchdog.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wan22-watchdog.timer
sudo systemctl start wan22-watchdog.service   # 立刻跑一次试
```

正常不发。会盯：主机内存 ≥85%、`/` `/data` `/opt` 磁盘 ≥85%、worker 不是 `active`、`nvidia-smi` 挂了、GPU 温度 ≥85°C。同类告警默认 30 分钟冷却。

**不按显存占用百分比告警**（Wan 常驻 48GB 卡上占用高是正常的）。

只测飞书通不通（会往群里打一张绿色测试卡片）：

```bash
sudo bash /opt/server/ops/test-feishu.sh
```

SSH 用户要能 `journalctl -u wan22-gpu`：`sudo usermod -aG systemd-journal ubuntu` 后重新登录。

## 2. Mac：抽取脚本

```bash
cd server/ops
cp .env.example .env
# 填 WAN22_GPU_SSH、WAN22_FEISHU_WEBHOOK、单元名
chmod +x hourly.sh daily.sh watchdog.sh
ssh -o BatchMode=yes "$WAN22_GPU_SSH" 'journalctl -u wan22-gpu -n 5 --no-pager'

./hourly.sh --notify    # 这一小时没问题则无输出、不发飞书
./daily.sh --notify     # 昨天 08:00 到今天 08:00（上海），必发
```

日报窗口按 **Asia/Shanghai 08:00–08:00**，SSH 到 GPU 时用 UTC 绝对时间，不依赖 GPU 时区。

## 3. Hermes cron

网关用 launchd 保活，合盖不要睡。任务里写绝对路径，**不要再让 Hermes 往飞书送一遍**（脚本已经 `--notify`）。

小时（每小时）：

```text
Run this exact command, do not interpret logs yourself:
/Users/YOU/Workspace/Cloned/h3/server/ops/hourly.sh --notify
If it exits 0, you are done. Do not send another Feishu message.
```

每天 8:00 Asia/Shanghai：

```text
Run this exact command, do not recount numbers:
/Users/YOU/Workspace/Cloned/h3/server/ops/daily.sh --notify
If it exits 0, you are done. Do not send another Feishu message.
```

Hermes cron 不稳时，用 launchd / crontab 直接跑这两条即可，不依赖 agent。

## 飞书

群自定义机器人，消息是 **卡片**（`interactive`）：日报绿头、小时告警橙头、watchdog 红头。签名算法见 `feishu.py`。
