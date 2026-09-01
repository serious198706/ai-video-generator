内网图生视频，无鉴权。Base URL 用 **API Gateway**（不要再打 GPU `:8000`）。

可空字段 JSON 里是 `null`（不要用缺 key 判断）。数字不要加引号。`POST /v1/generate` **只校验 URL 并入 Redis 队列，不下图**，超时可以比 30s 短。单卡串行，不要同步死等成片。

---

提交任务

请求
- url: `/v1/generate`
- method: POST
- 请求参数:
Content-Type: application/json

字段说明：

| 字段名 | 类型 | 必须 | 说明 |
| --- | --- | --- | --- |
| image | string | 是 | 首帧 HTTPS URL，公网图（已加白名单的 CloudFront 等）。接口只校验，GPU 出队后再下载。不要传 base64 / multipart |
| prompt | string | 否 | 空或省略则用服务端默认提示词 |
| negativePrompt | string | 否 | 会传给模型 |
| duration | number | 否 | 秒，`(0, 15]`，默认 5。可写 5 或 5.0 |
| resolution | string | 否 | 仅 `540p` / `720p` / `1080p`。本轮只记录，画布仍约 480×832，不会真超分 |
| webhookUrl | string | 否 | 成功、失败都会 POST。必须 https，host 需在 `WAN22_WEBHOOK_HOSTS`（允许内网） |
| steps | integer | 否 | 1–50，不传则用服务端默认 |
| quality | integer | 否 | 导出质量 1–10 |
| seed | integer | 否 | 不传则服务端随机；完成后可在查询 / webhook 拿到实际值 |
| lastImage | string | 否 | 尾帧 HTTPS URL，约束同 image |
| audio | boolean | 否 | 是否配 Foley。默认 `true`。`false` 则只出无声片，不占 Foley 时间。服务端关掉 Foley 时即使传 `true` 也是无声片 |

请求示例：

```json
{
  "image": "https://d2ud1xskuh00y0.cloudfront.net/test1.jpg",
  "prompt": "nsfwsks, she slowly turns her head",
  "negativePrompt": "blurry",
  "duration": 5,
  "resolution": "540p",
  "webhookUrl": "https://<host>/internal/wan/callback",
  "steps": 4,
  "quality": 6,
  "seed": 123,
  "lastImage": null,
  "audio": true
}
```

返回
- HTTP 202（此时只是已入队，不是成片完成）
- id 与 task_id 相同，用来轮询

字段说明：

| 字段名 | 类型 | 必须 | 说明 |
| --- | --- | --- | --- |
| id | string | 是 | 32 位 hex（无连字符 UUID） |
| task_id | string | 是 | 与 id 相同 |
| status | string | 是 | 恒为 `queued` |

返回示例：

```json
{
  "id": "d4d4c0ea896e4f6ca6425930c41398aa",
  "task_id": "d4d4c0ea896e4f6ca6425930c41398aa",
  "status": "queued"
}
```

错误（body 为 `{"detail": "..."}`；422 时 detail 为数组）：

| HTTP | 说明 |
| --- | --- |
| 400 | 图 / webhook URL 不合法：非 https、host 不在白名单、图解析到私网 |
| 422 | 字段类型或范围不对（duration 超 15、resolution 不是那三个枚举等） |
| 429 | 排队 List ≥ 500（正在跑的不算进这 500） |
| 503 | Redis 不可用或入队失败 |

---

查询任务

请求
- url: `/v1/tasks/{id}`
- method: GET
- 请求参数:

| 字段名 | 类型 | 必须 | 说明 |
| --- | --- | --- | --- |
| id | string | 是 | path 参数，提交时返回的 task_id |

返回
- HTTP 200
- 建议 3–5s 轮询，`succeeded` / `failed` 后停。有 webhook 时查询只作兜底

字段说明：

| 字段名 | 类型 | 必须 | 说明 |
| --- | --- | --- | --- |
| id | string | 是 | 与 task_id 相同 |
| task_id | string | 是 | |
| status | string | 是 | `queued` / `running` / `succeeded` / `failed` |
| prompt | string | 是 | 实际使用的提示词 |
| duration | number | 是 | 秒，如 5.0 |
| resolution | string | 否 | `540p` / `720p` / `1080p`，未传为 null |
| seed | integer | 否 | 未指定且未跑完为 null；成功后为实际种子 |
| video_url | string | 否 | 成功为 CloudFront mp4；其它状态 null |
| error | string | 否 | 仅 failed 时有短码；其它状态 null |
| audio | boolean | 是 | 这条任务是否要求配音 |
| created_at | string | 是 | UTC ISO-8601 |
| updated_at | string | 是 | UTC ISO-8601 |

status 与成片字段：

| status | 含义 | video_url | error |
| --- | --- | --- | --- |
| queued | 排队 | null | null |
| running | 正在生成 | null | null |
| succeeded | 成片已上传 | CloudFront mp4 | null |
| failed | 失败（终态） | null | 短码 |

error 短码（不要当给人看的长文案）：

| 值 | 说明 |
| --- | --- |
| generate_failed | 推理失败（含重试仍失败） |
| foley_failed | 成片后配 Foley 失败（仅 `WAN22_FOLEY_REQUIRED=1`） |
| upload_failed | 成片上传 S3 失败 |
| download_failed | GPU 下图失败 |
| interrupted | 进程重启后重试仍失败（最多 3 次，含首次） |

返回示例：

```json
{
  "id": "d4d4c0ea896e4f6ca6425930c41398aa",
  "task_id": "d4d4c0ea896e4f6ca6425930c41398aa",
  "status": "succeeded",
  "prompt": "nsfwsks, she slowly turns her head",
  "duration": 5.0,
  "resolution": "540p",
  "seed": 123,
  "video_url": "https://d2ud1xskuh00y0.cloudfront.net/video/d4d4c0ea896e4f6ca6425930c41398aa.mp4",
  "error": null,
  "audio": true,
  "created_at": "2026-08-27T09:26:00.000000+00:00",
  "updated_at": "2026-08-27T09:26:41.000000+00:00"
}
```

错误：

| HTTP | 说明 |
| --- | --- |
| 404 | 任务不存在 |
| 503 | 队列不可用 |

---

Webhook（可选）

请求（GPU 调 Java）
- url: 提交时的 webhookUrl
- method: POST
- 请求参数:
Content-Type: application/json

任务进入 `succeeded` 或 `failed` 时回调（失败也会 POST）。期望返回 2xx。超时约 5s，最多重试 3 次。必须 https；host 需在 `WAN22_WEBHOOK_HOSTS`。图床域名需在 `WAN22_IMAGE_HOSTS`。

Header：

| 字段名 | 类型 | 必须 | 说明 |
| --- | --- | --- | --- |
| X-Wan-Signature | string | 否 | 仅当 GPU 配置了 WAN22_WEBHOOK_SECRET。值为 `sha256=<hex>`，对原始 body 字节做 HMAC-SHA256 |

字段说明：

| 字段名 | 类型 | 必须 | 说明 |
| --- | --- | --- | --- |
| id | string | 是 | 与 task_id 相同 |
| task_id | string | 是 | |
| status | string | 是 | `succeeded` 或 `failed` |
| video_url | string | 否 | 成功为 mp4 URL，失败为 null |
| error | string | 否 | 失败为短码，成功为 null |
| seed | integer | 否 | |
| duration | number | 否 | 秒 |
| resolution | string | 否 | |

请求示例：

```json
{
  "id": "d4d4c0ea896e4f6ca6425930c41398aa",
  "task_id": "d4d4c0ea896e4f6ca6425930c41398aa",
  "status": "succeeded",
  "video_url": "https://d2ud1xskuh00y0.cloudfront.net/video/d4d4c0ea896e4f6ca6425930c41398aa.mp4",
  "error": null,
  "seed": 123,
  "duration": 5.0,
  "resolution": "540p"
}
```

---

探活

健康检查（API Gateway / Lambda）

请求
- url: `/health`
- method: GET

返回
- HTTP 200：能连上 Redis
- HTTP 503：Redis 不可用

字段说明（200）：

| 字段名 | 类型 | 必须 | 说明 |
| --- | --- | --- | --- |
| ok | boolean | 是 | |

GPU 本机还有 `/health`、`/ready`（模型是否已加载），**不要走 Java 主路径**，只给运维。
