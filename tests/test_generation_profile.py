from __future__ import annotations

import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("WAN22_DRY_RUN", "1")
os.environ.setdefault("WAN22_PRELOAD", "0")

from wan22 import config
from wan22.infer import generate


class GenerationProfileTests(unittest.TestCase):
    def setUp(self):
        self.torch = Mock()
        self.torch.cuda.get_device_properties.return_value.total_memory = 96 * 1024**3
        self.torch.cuda.get_device_capability.return_value = (12, 0)
        self.torch.cuda.mem_get_info.return_value = (40 * 1024**3, 96 * 1024**3)
        self.torch.cuda.max_memory_allocated.return_value = 30 * 1024**3
        self.torch.cuda.max_memory_reserved.return_value = 40 * 1024**3
        self.modules = patch.dict(sys.modules, {"torch": self.torch})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_auto_residency_uses_load_and_actual_headroom(self):
        with patch.object(config, "EXPERT_RESIDENCY", "auto"), patch.object(config, "OFFLOAD", "none"):
            # Same 720p canvas for 720p and upscaled 1080p; output label is irrelevant.
            self.assertTrue(generate._resident_experts(720, 1248, 81))
            self.assertTrue(generate._resident_experts(480, 832, 237))
            self.assertFalse(generate._resident_experts(720, 1248, 237))
            self.torch.cuda.mem_get_info.return_value = (64 * 1024**3, 96 * 1024**3)
            self.assertTrue(generate._resident_experts(720, 1248, 237))
            self.torch.cuda.mem_get_info.return_value = (8 * 1024**3, 96 * 1024**3)
            self.assertFalse(generate._resident_experts(480, 832, 77))
            self.torch.cuda.get_device_properties.return_value.total_memory = 48 * 1024**3
            self.assertFalse(generate._resident_experts(480, 832, 77))

    def test_explicit_off_and_offload_keep_original_policy(self):
        with patch.object(config, "EXPERT_RESIDENCY", "off"), patch.object(config, "OFFLOAD", "none"):
            self.assertFalse(generate._resident_experts(720, 1248, 237))
        with patch.object(config, "EXPERT_RESIDENCY", "on"), patch.object(config, "OFFLOAD", "model"):
            self.assertFalse(generate._resident_experts(720, 1248, 237))
            pipe = Mock()
            self.assertIsNone(generate._stage_experts(pipe, 237))
            pipe.transformer.to.assert_not_called()
        with patch.object(config, "EXPERT_RESIDENCY", "on"), patch.object(config, "OFFLOAD", "none"):
            self.assertTrue(generate._resident_experts(720, 1248, 237))

    def test_staging_offloads_before_loading_and_does_not_repeat_within_expert(self):
        events = Mock()
        pipe = SimpleNamespace(
            transformer=events.high, transformer_2=events.low,
            config=SimpleNamespace(boundary_ratio=0.9),
            scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000), timesteps=[999, 950, 800, 500]),
        )
        callback = generate._stage_experts(pipe, 237)
        self.assertEqual([(c[0], c.args) for c in events.mock_calls], [("low.to", ("cpu",)), ("high.to", ("cuda",))])
        events.reset_mock()
        self.torch.cuda.empty_cache.reset_mock()
        callback(pipe, 0, 999, {})
        self.assertEqual(events.mock_calls, [])
        self.torch.cuda.empty_cache.assert_not_called()
        callback(pipe, 1, 950, {})
        self.assertEqual([(c[0], c.args) for c in events.mock_calls], [("high.to", ("cpu",)), ("low.to", ("cuda",))])
        events.reset_mock()
        callback(pipe, 2, 800, {})
        self.assertEqual(events.mock_calls, [])

    def test_resident_experts_have_no_swapping_callback(self):
        pipe = SimpleNamespace(transformer=Mock(), transformer_2=Mock())
        self.assertIsNone(generate._stage_experts(pipe, 237, resident=True))
        pipe.transformer.to.assert_called_once_with("cuda")
        pipe.transformer_2.to.assert_called_once_with("cuda")

    def test_original_long_video_policy_still_swaps_at_boundary(self):
        pipe = SimpleNamespace(
            transformer=Mock(), transformer_2=Mock(),
            config=SimpleNamespace(boundary_ratio=0.9),
            scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000), timesteps=[999, 800]),
        )
        callback = generate._stage_experts(pipe, 237)
        pipe.transformer_2.to.assert_called_with("cpu")
        payload = {"latents": object()}
        self.assertIs(callback(pipe, 0, 999, payload), payload)
        pipe.transformer_2.to.assert_called_with("cuda")
        pipe.transformer.to.assert_called_with("cpu")

    def test_profile_accumulates_and_restores_methods_after_error(self):
        class Component:
            def forward(self):
                return 7

            def decode(self):
                raise RuntimeError("decode failed")

        pipe = SimpleNamespace(
            encode_prompt=lambda: "prompt", prepare_latents=lambda: "latents",
            transformer=Component(), transformer_2=Component(), vae=Component(),
        )
        original_prompt = pipe.encode_prompt
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            with generate._profile_generation(pipe) as stats:
                self.assertEqual(pipe.encode_prompt(), "prompt")
                self.assertEqual(pipe.prepare_latents(), "latents")
                self.assertEqual(pipe.transformer.forward(), 7)
                self.assertEqual(pipe.transformer_2.forward(), 7)
                pipe.vae.decode()
        self.assertIs(pipe.encode_prompt, original_prompt)
        self.assertNotIn("forward", vars(pipe.transformer))
        self.assertNotIn("forward", vars(pipe.transformer_2))
        self.assertNotIn("decode", vars(pipe.vae))
        self.assertEqual(set(stats), {"prompt_s", "latent_prepare_s", "denoise_s", "denoise_high_s", "denoise_low_s", "vae_decode_s", "peak_allocated_gib", "peak_reserved_gib"})
        self.assertEqual(stats["denoise_s"], stats["denoise_high_s"] + stats["denoise_low_s"])
        self.assertTrue(all(value >= 0 for value in stats.values()))

    def test_generation_profiles_only_1080p_and_keeps_request_parameters(self):
        # 同一 pipeline 连续处理不同分辨率，包装不可泄漏到后续请求。
        class Pipe:
            def __init__(self):
                self.transformer = SimpleNamespace(forward=lambda: None)
                self.transformer_2 = SimpleNamespace(forward=lambda: None)
                self.vae = SimpleNamespace(decode=lambda: None)
                self.calls = []

            def encode_prompt(self):
                pass

            def prepare_latents(self):
                pass

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                self.encode_prompt()
                self.prepare_latents()
                self.transformer.forward()
                self.vae.decode()
                return SimpleNamespace(frames=[[[0.5]]])

        pipe = Pipe()
        export = Mock()
        image = SimpleNamespace(width=720, height=1248)
        with ExitStack() as stack:
            stack.enter_context(patch.object(config, "DRY_RUN", False))
            stack.enter_context(patch.object(config, "PROFILE", "auto"))
            stack.enter_context(patch("wan22.infer.attention.snapshot", return_value={}))
            stack.enter_context(patch("wan22.infer.attention.delta", return_value={}))
            self.torch.cuda.mem_get_info.return_value = (64 * 1024**3, 96 * 1024**3)
            stack.enter_context(patch.object(config, "EXPERT_RESIDENCY", "auto"))
            stack.enter_context(patch.object(config, "OFFLOAD", "none"))
            stack.enter_context(patch.object(config, "QUANT_1080P", "baseline"))
            stack.enter_context(patch.dict(sys.modules, {"diffusers.utils": SimpleNamespace(export_to_video=export)}))
            stack.enter_context(patch.object(generate, "load_pipe", return_value=pipe))
            stack.enter_context(patch("wan22.media.image.open_rgb", return_value=image))
            stack.enter_context(patch.object(generate, "_resize_for_wan", return_value=image))
            stage = stack.enter_context(patch.object(generate, "_stage_experts", return_value=None))
            for resolution in ("1080p", "720p", "480p"):
                seed = generate.generate_video(
                    prompt="test", output_path="unused.mp4", first_frame_path="unused.jpg",
                    last_frame_path=None, duration=15, seed=42, steps=4, resolution=resolution,
                )
                self.assertEqual(seed, 42)
                self.assertNotIn("encode_prompt", vars(pipe))
                self.assertNotIn("prepare_latents", vars(pipe))
            self.assertEqual([call.kwargs["resident"] for call in stage.call_args_list], [True, True, True])
        self.torch.cuda.reset_peak_memory_stats.assert_called_once()
        self.assertEqual(export.call_count, 3)
        for request in pipe.calls:
            self.assertEqual(request["num_inference_steps"], 4)
            self.assertEqual(request["num_frames"], 237)

    def test_auto_fp8_is_disabled_even_on_large_blackwell_gpu(self):
        with patch.object(config, "QUANT_1080P", "auto"), patch.object(config, "OFFLOAD", "none"):
            self.assertFalse(generate._use_fp8_1080p("1080p"))
            for resolution in (None, "480p", "720p"):
                self.assertFalse(generate._use_fp8_1080p(resolution))
            self.torch.cuda.get_device_properties.return_value.total_memory = 48 * 1024**3
            self.assertFalse(generate._use_fp8_1080p("1080p"))
            self.torch.cuda.get_device_capability.return_value = (8, 0)
            self.assertFalse(generate._use_fp8_1080p("1080p"))

    def test_fp8_cache_restores_original_experts_without_requantizing(self):
        baseline = (Mock(), Mock())
        fp8 = (Mock(), Mock())
        pipe = SimpleNamespace(transformer=baseline[0], transformer_2=baseline[1])
        quantize = Mock()
        fake_ao = SimpleNamespace(Float8DynamicActivationFloat8WeightConfig=Mock(), quantize_=quantize)
        with (
            patch.object(config, "QUANT_1080P", "float8dq"),
            patch.object(config, "OFFLOAD", "none"),
            patch.object(generate, "_load_transformer", side_effect=fp8) as load,
            patch.object(generate, "_prepare_fp8_linears", return_value=lambda *_: True),
            patch.object(generate, "_fuse_nsfw_loras") as fuse,
            patch.dict(sys.modules, {"torchao.quantization": fake_ao, "torchao.quantization.granularity": SimpleNamespace(PerRow=Mock())}),
        ):
            generate._select_transformers(pipe, "1080p")
            self.assertIs(pipe.transformer, fp8[0])
            self.assertEqual(load.call_count, 2)
            fuse.assert_called_once_with(pipe)
            self.assertEqual([call.args[0] for call in quantize.call_args_list], list(fp8))
            generate._select_transformers(pipe, "1080p")
            self.assertEqual(load.call_count, 2)
            generate._select_transformers(pipe, "480p")
            self.assertIs(pipe.transformer, baseline[0])
            self.assertIs(pipe.transformer_2, baseline[1])
            generate._select_transformers(pipe, "1080p")
            self.assertIs(pipe.transformer, fp8[0])
            self.assertEqual(load.call_count, 2)
            self.assertEqual(quantize.call_count, 2)

    def test_failed_fp8_build_restores_baseline_pair(self):
        baseline = (Mock(), Mock())
        pipe = SimpleNamespace(transformer=baseline[0], transformer_2=baseline[1])
        fake_ao = SimpleNamespace(Float8DynamicActivationFloat8WeightConfig=Mock(), quantize_=Mock(side_effect=RuntimeError("unsupported kernel")))
        with (
            patch.object(config, "QUANT_1080P", "float8dq"),
            patch.object(generate, "_use_fp8_1080p", return_value=True),
            patch.object(generate, "_load_transformer", side_effect=[Mock(), Mock()]),
            patch.object(generate, "_prepare_fp8_linears", return_value=lambda *_: True),
            patch.object(generate, "_fuse_nsfw_loras"),
            patch.dict(sys.modules, {"torchao.quantization": fake_ao, "torchao.quantization.granularity": SimpleNamespace(PerRow=Mock())}),
        ):
            with self.assertRaisesRegex(RuntimeError, "unsupported kernel"):
                generate._select_transformers(pipe, "1080p")
        self.assertIs(pipe.transformer, baseline[0])
        self.assertIs(pipe.transformer_2, baseline[1])
        self.assertEqual(pipe._wan22_expert_mode, "baseline")
        self.assertEqual(set(pipe._wan22_expert_cache), {"baseline"})

    def test_auto_fp8_failure_continues_on_baseline_and_does_not_rebuild(self):
        baseline = (Mock(), Mock())
        pipe = SimpleNamespace(transformer=baseline[0], transformer_2=baseline[1])
        fake_ao = SimpleNamespace(Float8DynamicActivationFloat8WeightConfig=Mock(), quantize_=Mock(side_effect=AssertionError("PerRow requires BF16")))
        with (
            patch.object(config, "QUANT_1080P", "auto"),
            patch.object(generate, "_use_fp8_1080p", return_value=True),
            patch.object(generate, "_load_transformer", side_effect=[Mock(), Mock()]) as load,
            patch.object(generate, "_prepare_fp8_linears", return_value=lambda *_: True),
            patch.object(generate, "_fuse_nsfw_loras"),
            patch.dict(sys.modules, {"torchao.quantization": fake_ao, "torchao.quantization.granularity": SimpleNamespace(PerRow=Mock())}),
        ):
            self.assertEqual(generate._select_transformers(pipe, "1080p"), config.QUANT)
            self.assertTrue(pipe._wan22_fp8_disabled)
            self.assertEqual(generate._select_transformers(pipe, "1080p"), config.QUANT)
            self.assertEqual(load.call_count, 2)
        self.assertIs(pipe.transformer, baseline[0])
        self.assertIs(pipe.transformer_2, baseline[1])
        self.assertEqual(set(pipe._wan22_expert_cache), {"baseline"})

    def test_sharing_requires_real_free_memory_and_only_affects_1080p(self):
        with patch.object(config, "UPSCALE_KEEP_WAN_1080P", "auto"), patch.object(config, "OFFLOAD", "none"):
            self.assertTrue(generate.can_share_gpu_for_upscale("1080p"))
            self.assertFalse(generate.can_share_gpu_for_upscale("720p"))
            self.torch.cuda.mem_get_info.return_value = (8 * 1024**3, 96 * 1024**3)
            self.assertFalse(generate.can_share_gpu_for_upscale("1080p"))

    def test_initial_gpu_preparation_checks_capacity_and_free_memory(self):
        pipe = Mock()
        with patch.object(config, "OFFLOAD", "none"):
            self.assertFalse(generate._prepare_initial_gpu(pipe))
            pipe.to.assert_not_called()
            self.torch.cuda.mem_get_info.return_value = (85 * 1024**3, 96 * 1024**3)
            self.assertTrue(generate._prepare_initial_gpu(pipe))
            pipe.to.assert_called_once_with("cuda")
            pipe.to.reset_mock()
            self.torch.cuda.get_device_properties.return_value.total_memory = 48 * 1024**3
            self.assertFalse(generate._prepare_initial_gpu(pipe))
            pipe.to.assert_not_called()

    def test_initial_gpu_oom_returns_pipeline_to_cpu(self):
        class Oom(RuntimeError):
            pass

        self.torch.cuda.OutOfMemoryError = Oom
        self.torch.cuda.mem_get_info.return_value = (85 * 1024**3, 96 * 1024**3)
        pipe = Mock()
        pipe.to.side_effect = [Oom("out of memory"), None]
        with patch.object(config, "OFFLOAD", "none"):
            self.assertFalse(generate._prepare_initial_gpu(pipe))
        self.assertEqual([call.args[0] for call in pipe.to.call_args_list], ["cuda", "cpu"])

    def test_failed_restore_invalidates_pipeline_and_releases_remaining_gpu_weights(self):
        pipe = Mock()
        pipe.to.side_effect = [RuntimeError("restore OOM"), None]
        with patch.object(generate, "_pipe", pipe), patch.object(generate, "_load_error", None), patch.object(config, "DRY_RUN", False):
            with self.assertRaisesRegex(RuntimeError, "restore OOM"):
                generate.resume_gpu()
            self.assertIsNone(generate._pipe)
            self.assertFalse(generate.is_ready())
            self.assertIn("restore OOM", generate.load_error())
        self.assertEqual([call.args[0] for call in pipe.to.call_args_list], ["cuda", "cpu"])


class VideoValidationTests(unittest.TestCase):
    def test_nonfinite_frames_and_entirely_black_video_are_rejected(self):
        import numpy as np

        for value in (np.nan, np.inf, -np.inf, 0.0):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invalid generated video"):
                generate._validate_video_frames(np.full((2, 2, 2, 3), value))
        with self.assertRaisesRegex(ValueError, "no frames"):
            generate._validate_video_frames([])

    def test_valid_video_with_one_black_frame_is_allowed(self):
        import numpy as np

        frames = np.zeros((2, 2, 2, 3))
        frames[1] = 0.5
        generate._validate_video_frames(frames)


class Fp8DtypeRegressionTests(unittest.TestCase):
    def test_lora_promoted_real_weights_are_bf16_before_perrow_quantization(self):
        import torch

        class CpuExpert(torch.nn.Module):
            _keep_in_fp32_modules = ["time_embedder"]

            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
                self.time_embedder = torch.nn.Linear(8, 8, dtype=torch.float32)

            def to(self, *args, **kwargs):
                # CPU 测试真实 tensor dtype；仅把设备迁移映射到 CPU。
                if args and args[0] == "cuda":
                    args = ("cpu", *args[1:])
                if kwargs.get("device") == "cuda":
                    kwargs["device"] = "cpu"
                return super().to(*args, **kwargs)

        baseline = (CpuExpert(), CpuExpert())
        fp8 = (CpuExpert(), CpuExpert())
        pipe = SimpleNamespace(transformer=baseline[0], transformer_2=baseline[1])

        def fuse(target):
            # 复现融合后 FP32 权重，而非从始至终使用不会检查 dtype 的 Mock。
            target.transformer.linear.float()
            target.transformer_2.linear.float()

        def quantize(expert, _config, *, filter_fn):
            self.assertEqual(expert.linear.weight.dtype, torch.bfloat16)
            self.assertTrue(filter_fn(expert.linear, "linear"))
            self.assertFalse(filter_fn(expert.time_embedder, "time_embedder"))
            self.assertEqual(expert.time_embedder.weight.dtype, torch.float32)

        fake_ao = SimpleNamespace(Float8DynamicActivationFloat8WeightConfig=Mock(), quantize_=Mock(side_effect=quantize))
        with (
            patch.object(config, "QUANT_1080P", "float8dq"),
            patch.object(generate, "_use_fp8_1080p", return_value=True),
            patch.object(generate, "_load_transformer", side_effect=fp8),
            patch.object(generate, "_fuse_nsfw_loras", side_effect=fuse),
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "empty_cache"),
            patch.dict(sys.modules, {"torchao.quantization": fake_ao, "torchao.quantization.granularity": SimpleNamespace(PerRow=Mock())}),
        ):
            self.assertEqual(generate._select_transformers(pipe, "1080p"), "float8dq")
        self.assertEqual(fake_ao.quantize_.call_count, 2)
        self.assertIs(pipe.transformer, fp8[0])
        self.assertEqual(baseline[0].linear.weight.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
