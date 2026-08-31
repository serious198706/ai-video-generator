"""HunyuanVideo-Foley 常驻进程。必须用 Foley 自己的 venv 启动，不要 import wan22。

stdin/stdout 走 JSON 行：
  -> {"cmd":"generate","video":"...","prompt":"...","neg_prompt":"...","wav":"...","steps":50,"guidance":4.5}
  <- {"ok":true,"wav":"..."} 或 {"ok":false,"error":"..."}
  -> {"cmd":"ping"} / {"cmd":"quit"}
日志只写 stderr。
"""

from __future__ import annotations

import os
import sys
import types


def _compat_py314() -> None:
    """Python 3.14 会拒绝 protobuf UPB 的 custom tp_new。

    环境变量不够：protobuf 4.x 先 import google._upb._message，失败是 TypeError
    不是 ImportError，纯 Python 实现根本轮不到。推理也不需要 tensorboard。
    """
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    class _BlockUpb:
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"google._upb", "google._upb._message"}:
                raise ImportError("protobuf UPB is incompatible with Python 3.14")
            return None

    sys.meta_path.insert(0, _BlockUpb())

    dummy = types.ModuleType("torch.utils.tensorboard")

    class _Writer:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    dummy.SummaryWriter = _Writer
    dummy.FileWriter = _Writer
    sys.modules["torch.utils.tensorboard"] = dummy
    sys.modules["torch.utils.tensorboard.writer"] = dummy


_compat_py314()

import argparse
import json
import traceback

import torch
import torch.utils as _torch_utils

_torch_utils.tensorboard = sys.modules["torch.utils.tensorboard"]


_NN_KEYS = (
    "foley_model",
    "dac_model",
    "siglip2_model",
    "clap_model",
    "syncformer_model",
)

_model_dict = None
_cfg = None
_cuda = None


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _modules(model_dict):
    for key in _NN_KEYS:
        model = getattr(model_dict, key, None)
        if model is not None and hasattr(model, "to"):
            yield key, model


def _to_device(model_dict, device) -> None:
    model_dict.device = device
    for _, model in _modules(model_dict):
        model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load(args) -> None:
    global _model_dict, _cfg, _cuda
    from hunyuanvideo_foley.utils.model_utils import load_model

    _cuda = torch.device("cuda:0")
    _log(f"loading foley size={args.model_size} path={args.model_path}")
    _model_dict, _cfg = load_model(
        args.model_path,
        args.config_path,
        _cuda,
        enable_offload=False,
        model_size=args.model_size,
    )
    _to_device(_model_dict, torch.device("cpu"))
    _log("foley ready on cpu")


def _generate(req: dict) -> dict:
    from hunyuanvideo_foley.utils.feature_utils import feature_process
    from hunyuanvideo_foley.utils.model_utils import denoise_process
    import torchaudio

    video = req["video"]
    wav = req["wav"]
    prompt = req.get("prompt") or ""
    neg = req.get("neg_prompt")
    steps = int(req.get("steps") or 50)
    guidance = float(req.get("guidance") or 4.5)

    _to_device(_model_dict, _cuda)
    try:
        visual_feats, text_feats, audio_len_in_s = feature_process(
            video,
            prompt,
            _model_dict,
            _cfg,
            neg_prompt=neg,
        )
        audio, sample_rate = denoise_process(
            visual_feats,
            text_feats,
            audio_len_in_s,
            _model_dict,
            _cfg,
            guidance_scale=guidance,
            num_inference_steps=steps,
        )
        torchaudio.save(wav, audio[0], sample_rate)
    finally:
        _to_device(_model_dict, torch.device("cpu"))
    return {"ok": True, "wav": wav}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--model-size", default="xl", choices=("xl", "xxl"))
    args = parser.parse_args()
    try:
        _load(args)
    except Exception as exc:
        _log(f"load failed: {exc}")
        traceback.print_exc(file=sys.stderr)
        try:
            _reply({"ok": False, "ready": False, "error": str(exc)})
        except Exception:
            pass
        os._exit(1)
    _reply({"ok": True, "ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "quit":
                _reply({"ok": True, "bye": True})
                return 0
            if cmd == "ping":
                _reply({"ok": True, "pong": True})
                continue
            if cmd != "generate":
                _reply({"ok": False, "error": f"unknown cmd {cmd!r}"})
                continue
            _reply(_generate(req))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _reply({"ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
