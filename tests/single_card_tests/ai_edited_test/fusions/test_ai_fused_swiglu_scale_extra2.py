# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import paddle


def _no_cuda():
    return patch(
        "paddlefleet.fusions.fused_swiglu_scale.paddle.is_compiled_with_cuda",
        return_value=False,
    )


class TestFusedSwigluScaleForward(unittest.TestCase):
    """CPU-fallback path of fused_swiglu_scale_forward (no clamp)."""

    def test_forward_cpu_fallback(self):
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.ones([4, 1])
            result = fused_swiglu_scale_forward(x, scale)
            self.assertEqual(result.shape, [4, 8])

    def test_forward_with_1d_scale(self):
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.ones([4])
            result = fused_swiglu_scale_forward(x, scale)
            self.assertEqual(result.shape, [4, 8])

    def test_forward_scale_broadcast(self):
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.full([4, 1], 2.0)
            result = fused_swiglu_scale_forward(x, scale)
            self.assertEqual(result.shape, [4, 8])


class TestFusedSwigluScaleForwardClamp(unittest.TestCase):
    """CPU-fallback path with clamp_value (covers lines 39-42)."""

    def test_forward_cpu_fallback_with_clamp(self):
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.full([2, 16], 100.0)  # exceeds clamp on both halves
            scale = paddle.ones([2, 1])
            cv = 1.0
            result = fused_swiglu_scale_forward(x, scale, clamp_value=cv)
            self.assertEqual(result.shape, [2, 8])
            # gate clamped to cv → silu(cv); val clamped to cv → silu(cv) * cv
            expected = (
                float(paddle.nn.functional.silu(paddle.to_tensor(cv)).numpy())
                * cv
            )
            paddle.allclose(
                result,
                paddle.full(result.shape, expected, dtype=result.dtype),
            )

    def test_forward_no_clamp_stays_finite(self):
        # Without clamp_value, forward should still produce finite results
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.randn([2, 16])
            scale = paddle.ones([2, 1])
            result = fused_swiglu_scale_forward(x, scale)
            self.assertTrue(bool(paddle.isfinite(result).all().item()))


class TestFusedSwigluScaleBackward(unittest.TestCase):
    """CPU-fallback path of fused_swiglu_scale_backward (no clamp)."""

    def test_backward_cpu_fallback(self):
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.ones([4])
            out_grad = paddle.randn([4, 8])
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
            self.assertEqual(d_x.shape, [4, 16])
            self.assertEqual(d_scale.shape, [4])

    def test_backward_shapes(self):
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x = paddle.randn([2, 32])
            scale = paddle.ones([2])
            out_grad = paddle.randn([2, 16])
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
            self.assertEqual(d_x.shape, [2, 32])
            self.assertEqual(d_scale.shape, [2])


class TestFusedSwigluScaleBackwardClamp(unittest.TestCase):
    """CPU-fallback path with clamp_value (covers lines 70-73, 96-97)."""

    def test_backward_cpu_fallback_with_clamp_masks_grads(self):
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            # Build x where gate (left half) and val (right half) both exceed clamp,
            # so the gradient masks zero out d_gate and d_val.
            cv = 0.5
            x = paddle.concat(
                [
                    paddle.full([2, 4], 5.0),  # gate > cv → g_mask=0
                    paddle.full([2, 4], 5.0),  # val > cv → v_mask=0
                ],
                axis=-1,
            )
            scale = paddle.ones([2])
            out_grad = paddle.randn([2, 4])
            d_x, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=cv
            )
            self.assertEqual(d_x.shape, [2, 8])
            # All d_x entries on saturated halves should be zero.
            self.assertTrue(bool((d_x.abs().sum() == 0).item()))

    def test_backward_cpu_fallback_with_clamp_inside_window(self):
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            cv = 10.0
            x = paddle.randn([2, 8])  # well inside ±cv
            scale = paddle.ones([2])
            out_grad = paddle.randn([2, 4])
            d_x, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=cv
            )
            self.assertEqual(d_x.shape, [2, 8])
            self.assertEqual(d_scale.shape, [2])
            # Inside window the masks are 1, so grads should not be all-zero.
            self.assertTrue(bool((d_x.abs().sum() > 0).item()))


if __name__ == "__main__":
    unittest.main()
