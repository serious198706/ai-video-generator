"""Opt-in standalone diagnostic; stops generation after one actual expert block."""
from __future__ import annotations

import json
import time
from pathlib import Path


class _BlockComplete(BaseException):
    """Private control flow, deliberately not caught by inference retry handlers."""


class FirstBlockProfile:
    def __init__(self, pipe, output: Path, *, cuda=True):
        self.pipe, self.output, self.cuda = pipe, Path(output), cuda
        self.result = None
        self._restore = []
        self._done = _BlockComplete()

    def _replace(self, module, forward):
        # Restore the original instance dictionary, including class-bound forwards.
        self._restore.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
        module.forward = forward

    def __enter__(self):
        import torch

        try:
            count = 0
            for expert_name in ("transformer", "transformer_2"):
                expert = getattr(self.pipe, expert_name, None)
                if expert is None:
                    continue
                block = expert.blocks[0]
                # Nested ranges expose projections separately from attention kernels.
                for name, child in block.named_modules():
                    if not name:
                        continue
                    original = child.forward

                    def ranged(*args, _forward=original, _name=name, **kwargs):
                        with torch.profiler.record_function(f"wan_block/{_name}"):
                            return _forward(*args, **kwargs)

                    self._replace(child, ranged)
                original = block.forward

                def measured(*args, _forward=original, _name=expert_name, _block=block, **kwargs):
                    return self._measure(_name, _block, _forward, args, kwargs)

                self._replace(block, measured)
                count += 1
            if not count:
                raise ValueError("No Wan expert blocks found")
        except BaseException:
            self._cleanup()
            raise
        return self

    def _measure(self, expert_name, block, forward, args, kwargs):
        import torch

        def describe(value):
            if isinstance(value, torch.Tensor):
                return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
            if isinstance(value, (tuple, list)):
                return [describe(item) for item in value]
            if isinstance(value, dict):
                return {key: describe(item) for key, item in value.items()}
            return type(value).__name__

        inputs = {"args": describe(args), "kwargs": describe(kwargs)}
        weights = {name: type(module.weight).__name__ for name, module in block.named_modules()
                   if hasattr(module, "weight") and module.weight is not None}
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.profiler.profile(activities=activities, record_shapes=False, with_stack=False) as profile:
            with torch.profiler.record_function("wan_block"):
                forward(*args, **kwargs)
            if self.cuda:
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self.output.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(self.output / "block-trace.json"))
        rows = [{"name": event.key, "calls": event.count,
                 "self_cpu_ms": event.self_cpu_time_total / 1000,
                 "cpu_total_ms": event.cpu_time_total / 1000,
                 "self_device_ms": event.self_device_time_total / 1000,
                 "device_total_ms": event.device_time_total / 1000}
                for event in profile.key_averages()]
        rows.sort(key=lambda row: row["self_device_ms"] if self.cuda else row["self_cpu_ms"], reverse=True)
        cuda_events = sum(event.device_type == torch.autograd.DeviceType.CUDA for event in profile.events())
        result = {"expert": expert_name, "block_index": 0, "wall_s_with_profiler": elapsed,
                  "inputs": inputs, "weight_types": weights, "cuda_events": cuda_events,
                  "timing_valid": not self.cuda or cuda_events > 0, "operators": rows,
                  "note": "One cold block, not full-generation or warmed timing. Nested total times overlap; do not sum them."}
        (self.output / "block-profile.json").write_text(json.dumps(result, indent=2) + "\n")
        if self.cuda and not cuda_events:
            raise RuntimeError("Profiler captured no CUDA events; inspect CUPTI availability and block-trace.json")
        self.result = result
        raise self._done

    def _cleanup(self):
        for module, existed, value in reversed(self._restore):
            if existed:
                module.forward = value
            else:
                del module.forward
        self._restore.clear()

    def __exit__(self, exc_type, exc, traceback):
        self._cleanup()
        if exc is self._done:
            return True
        if exc is None:
            raise RuntimeError("Generation returned without executing a profiled block")
        return False
