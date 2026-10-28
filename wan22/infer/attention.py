"""Guarded Sage dispatch with per-job counters and a persistent failure circuit breaker."""
from __future__ import annotations

from collections import Counter
from importlib.metadata import PackageNotFoundError, version

from wan22.log import get_logger

logger = get_logger(__name__)


class SageDispatcher:
    _wan22_sage = True

    def __init__(self, original, sage, backend: str):
        self.original = original
        self.sage = sage
        self.backend = backend
        self.counts = Counter()
        self.disabled = set()

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        import torch

        # Do not silently ignore SDPA-only semantics (custom scale, GQA, masks).
        eligible = (
            query.is_cuda and query.ndim == key.ndim == value.ndim == 4
            and query.dtype in (torch.float16, torch.bfloat16)
            and key.dtype == value.dtype == query.dtype
            and attn_mask is None and not is_causal and dropout_p == 0.0
            and query.size(-1) <= 128
            and kwargs.get("scale") is None and not kwargs.get("enable_gqa", False)
            and not (set(kwargs) - {"scale", "enable_gqa"})
        )
        # Full shapes distinguish self/cross attention and unsupported lengths.
        signature = (str(query.device), query.dtype, tuple(query.shape), tuple(key.shape), tuple(value.shape))
        if eligible and signature not in self.disabled:
            try:
                result = self.sage(query, key, value, is_causal=False)
            except torch.cuda.OutOfMemoryError:
                # Trying SDPA on an exhausted device may worsen OOM; preserve the error.
                raise
            except Exception:
                self.disabled.add(signature)
                self.counts["failures"] += 1
                logger.exception("%s failed; disable this attention shape until restart", self.backend)
            else:
                self.counts["sage_calls"] += 1
                return result
        self.counts["sdpa_calls"] += 1
        if eligible:
            self.counts["fallback_calls"] += 1
        return self.original(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, **kwargs,
        )


def install(enabled: bool, backend: str) -> bool:
    import torch.nn.functional as functional

    current = functional.scaled_dot_product_attention
    original = current.original if isinstance(current, SageDispatcher) else current
    if not enabled or backend == "sdpa":
        functional.scaled_dot_product_attention = original
        logger.info("attention backend=sdpa")
        return False
    candidates = ("sage3", "sage2") if backend == "auto" else (backend,)
    for candidate in candidates:
        try:
            if candidate == "sage3":
                from sageattn3 import sageattn3_blackwell as sage
            else:
                import sageattention
                try:
                    installed = version("sageattention")
                except PackageNotFoundError:
                    installed = str(getattr(sageattention, "__version__", "0"))
                if int(installed.split(".")[0]) < 2:
                    raise ImportError(f"SageAttention >=2 required, got {installed}")
                sage = sageattention.sageattn
            functional.scaled_dot_product_attention = SageDispatcher(original, sage, candidate)
            logger.info("attention backend=%s installed; runtime counters verify actual use", candidate)
            return True
        except Exception:
            logger.exception("attention backend=%s unavailable", candidate)
    functional.scaled_dot_product_attention = original
    logger.warning("attention backend=sdpa (requested %s unavailable)", backend)
    return False


def snapshot() -> dict:
    import torch.nn.functional as functional

    dispatch = functional.scaled_dot_product_attention
    if not isinstance(dispatch, SageDispatcher):
        return {"backend": "sdpa", "instrumented": False}
    return {"backend": dispatch.backend, "instrumented": True, **dict(dispatch.counts)}


def delta(before: dict) -> dict:
    after = snapshot()
    result = {"backend": after["backend"], "instrumented": after["instrumented"]}
    for name in ("sage_calls", "sdpa_calls", "fallback_calls", "failures"):
        result[name] = after.get(name, 0) - before.get(name, 0)
    logger.info("attention %s", " ".join(f"{key}={value}" for key, value in result.items()))
    if result["instrumented"] and result["sage_calls"] == result["sdpa_calls"] == 0:
        logger.warning("Sage installed but no SDPA wrapper calls observed; verify Diffusers attention dispatch")
    return result
