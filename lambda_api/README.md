# Lambda 接单层（已停用）

当前接单走 GPU FastAPI：`uvicorn wan22.api.app:app` / `./start.sh`。不再用 Redis / ElastiCache。

这支 Lambda 只返回 503，避免误打 API Gateway。
