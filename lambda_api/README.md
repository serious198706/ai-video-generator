# Lambda SQS 接单层

API Gateway 调用此 Lambda；Lambda 校验与 GPU API 相同的生成请求后，将任务发到 SQS。Vast.ai worker（`server/wan22` 的 `sqs_consumer`，配 `SQS_QUEUE_URL` 即启用）主动消费 SQS，因此不需要 Vast.ai GPU 暴露随机 URL；凭证用注入的 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`（与 S3 上传共用身份），权限只需队列的 `ReceiveMessage` / `DeleteMessage` / `ChangeMessageVisibility`。

## 配置

设置 Lambda 环境变量：

- `SQS_QUEUE_URL`：目标 SQS 队列 URL
- `SQS_MAX_QUEUE`：队列达到此数量时拒绝新任务并返回 `429`；未设置或设为 `0` 表示不限制。该值基于 SQS 的近似可见消息数，存在短暂延迟，不能作为严格并发锁。

Lambda execution role 只需对目标队列授予 `sqs:SendMessage`：

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "sqs:SendMessage",
    "Resource": "arn:aws:sqs:REGION:ACCOUNT_ID:QUEUE_NAME"
  }]
}
```

## API

`POST /v1/generate` 接受与 GPU FastAPI 相同的 JSON 字段。入队成功返回 HTTP 202：

```json
{"id":"...","task_id":"...","status":"queued"}
```

SQS 消息包含 `task_id` 及生成参数：`prompt`、`negative_prompt`、`image_url`、`last_image_url`、`duration`、`resolution`、`webhook_url`、`seed`、`steps`、`quality`、`audio`。消费者应使用 `task_id` 做幂等处理，并在成功完成后删除消息。

API Gateway 应将该路径代理到 `lambda_api.handler.handler`。继续用 `./package.sh` 打包，运行时为 Python 3.12 / x86_64。
