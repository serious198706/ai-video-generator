# Lambda SQS 接单层

API Gateway 调用此 Lambda；Lambda 校验与 GPU API 相同的生成请求后，将任务发到 SQS。Vast.ai worker（`server/wan22` 的 `sqs_consumer`，配 `SQS_QUEUE_URL` 即启用）主动消费 SQS，因此不需要 Vast.ai GPU 暴露随机 URL；凭证用注入的 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`（与 S3 上传共用身份），权限只需队列的 `ReceiveMessage` / `DeleteMessage` / `ChangeMessageVisibility`。

## 配置

设置 Lambda 环境变量：

- `SQS_QUEUE_URL`：目标 SQS 队列 URL
- `SQS_MAX_QUEUE`：队列达到此数量时拒绝新任务并返回 `429`；未设置或设为 `0` 表示不限制。该值基于 SQS 的近似可见消息数，存在短暂延迟，不能作为严格并发锁。
- `S3_BUCKET`：任务状态文件所在的桶。默认 `generated-video-905641673118-us-east-1-an`；设为空则不写 `queued`、查询接口返回 503。
- `STATUS_PREFIX`：状态文件前缀，默认 `video/wan22/`，需与 GPU 侧 `WAN22_STATUS_PREFIX`（默认同 `WAN22_S3_PREFIX`）一致。实际 key 为 `{STATUS_PREFIX}/status/{task_id}.json`。
- `S3_ENDPOINT`：自建/本地 S3 兼容端点，线上留空。
- `STATUS_STALL_SECONDS`：`generating` 超过这么久没更新就报 `stalled`（默认 2700s，要大于最坏单次生成耗时）。设 `0` 关闭。

Lambda execution role 需要 SQS 发消息和状态对象读写：

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sqs:SendMessage",
      "Resource": "arn:aws:sqs:REGION:ACCOUNT_ID:QUEUE_NAME"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject"],
      "Resource": "arn:aws:s3:::generated-video-905641673118-us-east-1-an/video/wan22/status/*"
    }
  ]
}
```

## API

`POST /v1/generate` 接受与 GPU FastAPI 相同的 JSON 字段。入队成功返回 HTTP 202：

```json
{"id":"...","task_id":"...","status":"queued"}
```

SQS 消息包含 `task_id` 及生成参数：`prompt`、`negative_prompt`、`image_url`、`last_image_url`、`duration`、`resolution`、`webhook_url`、`seed`、`steps`、`quality`、`audio`。消费者用 `task_id` 做幂等，终态后删除消息。

### 查询任务状态

`GET /v1/tasks/{task_id}` 读 S3 状态文件，返回最近一次状态：

```json
{"id":"...","task_id":"...","status":"generating","updated_at":"2026-10-26T08:00:00+00:00","worker_id":"...","resolution":"720p"}
```

| status | 含义 | 来源 |
| --- | --- | --- |
| `queued` | 已入队，还没被 worker 取走 | Lambda 入队时写 |
| `pending` | worker 已取到消息、占住执行位，等排队的前一单跑完 | GPU worker |
| `generating` | 正在推理 | GPU worker |
| `succeeded` | 成片已上传，带 `video_url` / `seed` | GPU worker |
| `failed` | 终态失败，带短码 `error` | GPU worker |
| `stalled` | `generating` 超 `STATUS_STALL_SECONDS` 没更新，worker 疑似挂了 | Lambda 查询时判定 |
| `unknown` | 没有状态文件：从未提交过，或对象已被生命周期规则清掉 | Lambda 兜底 |

注意：`queued` 与 `unknown` 的区别只在 Lambda 写状态成功时存在；SQS 本身不提供按 `task_id` 反查 in-flight 消息的能力，所以"能查"完全依赖这个状态文件。S3 读失败返回 503，不会伪装成 `unknown`。

API Gateway 应将该路径代理到 `lambda_api.handler.handler`。继续用 `./package.sh` 打包，运行时为 Python 3.12 / x86_64。
