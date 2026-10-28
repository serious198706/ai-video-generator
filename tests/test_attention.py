from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("WAN22_DRY_RUN", "1")
os.environ.setdefault("WAN22_PRELOAD", "0")

import torch
from wan22.infer.attention import SageDispatcher, delta, install, snapshot


def query(shape=(1, 2, 8, 64), *, dtype=torch.bfloat16, cuda=True):
    return SimpleNamespace(is_cuda=cuda, dtype=dtype, shape=shape, ndim=len(shape), device="cuda:0", size=lambda i: shape[i])


class AttentionTests(unittest.TestCase):
    def test_counts_success_and_preserves_noneligible_sdpa_arguments(self):
        original, sage = Mock(return_value="native"), Mock(return_value="fast")
        dispatch = SageDispatcher(original, sage, "sage2")
        q = query()
        self.assertEqual(dispatch(q, q, q, scale=None, enable_gqa=False), "fast")
        for kwargs in ({"attn_mask": object()}, {"scale": 0.25}, {"enable_gqa": True}, {"is_causal": True}, {"dropout_p": 0.1}):
            self.assertEqual(dispatch(q, q, q, **kwargs), "native")
            for name, value in kwargs.items():
                self.assertEqual(original.call_args.kwargs[name], value)
        self.assertEqual(dispatch.counts["sage_calls"], 1)
        self.assertEqual(dispatch.counts["sdpa_calls"], 5)
        self.assertEqual(dispatch.counts["fallback_calls"], 0)

    def test_failed_shape_is_tried_once_but_other_shapes_still_use_sage(self):
        sage = Mock(side_effect=[RuntimeError("unsupported shape"), "fast"])
        dispatch = SageDispatcher(Mock(return_value="native"), sage, "sage3")
        q = query()
        for _ in range(3):
            self.assertEqual(dispatch(q, q, q), "native")
        other = query((1, 2, 12, 64))
        self.assertEqual(dispatch(other, other, other), "fast")
        self.assertEqual(sage.call_count, 2)
        self.assertEqual(dispatch.counts["failures"], 1)
        self.assertEqual(dispatch.counts["fallback_calls"], 3)

    def test_oom_is_propagated_without_attempting_sdpa(self):
        original = Mock()
        dispatch = SageDispatcher(original, Mock(side_effect=torch.cuda.OutOfMemoryError("OOM")), "sage3")
        q = query()
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            dispatch(q, q, q)
        original.assert_not_called()

    def test_real_cpu_tensors_are_unchanged_and_job_counters_are_deltas(self):
        import torch.nn.functional as functional

        original = functional.scaled_dot_product_attention
        dispatch = SageDispatcher(original, Mock(), "sage2")
        q = torch.randn(1, 2, 8, 64)
        expected = original(q, q, q)
        with patch.object(functional, "scaled_dot_product_attention", dispatch):
            for _ in range(2):
                before = snapshot()
                actual = functional.scaled_dot_product_attention(q, q, q)
                self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(delta(before)["sdpa_calls"], 1)
        dispatch.sage.assert_not_called()

    def test_explicit_sdpa_unwraps_previous_dispatcher(self):
        import torch.nn.functional as functional

        original = functional.scaled_dot_product_attention
        with patch.object(functional, "scaled_dot_product_attention", SageDispatcher(original, Mock(), "sage3")):
            self.assertFalse(install(True, "sdpa"))
            self.assertIs(functional.scaled_dot_product_attention, original)

    def test_explicit_sage2_does_not_select_sage3(self):
        import torch.nn.functional as functional

        with (
            patch.object(functional, "scaled_dot_product_attention", functional.scaled_dot_product_attention),
            patch.dict(sys.modules, {"sageattention": SimpleNamespace(sageattn=Mock())}),
            patch("wan22.infer.attention.version", return_value="2.2.0"),
        ):
            self.assertTrue(install(True, "sage2"))
            self.assertEqual(snapshot()["backend"], "sage2")


if __name__ == "__main__":
    unittest.main()
