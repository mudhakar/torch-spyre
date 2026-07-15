# Copyright 2026 The Torch-Spyre Authors.
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

"""End-to-end tests for matmul N-dimension span padding (issue #1918).

A matmul whose weight has a per-core span over the 256MB EAR limit can only be
made legal by splitting its outer stick dimension across cores. When the stick
count is prime (e.g. vocab 49216 -> 49216/64 = 769 sticks), span_reduction cannot
split it and the compile aborts. lower_mm pads the N dimension up to the nearest
composite stick count, then narrows the output back to the true N.

These tests compile+run on a Spyre device, so run them on the pod.
"""

import unittest
from typing import Any
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch._inductor import config as t_inductor_config
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Reduction

from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor import passes
from torch_spyre._inductor.constants import BATCH_MATMUL_OP
from torch_spyre._inductor.passes import CustomPreSchedulingPasses

HIDDEN = 4096


class _CapturePasses(CustomPreSchedulingPasses):
    """Capture the operations list after all pre-scheduling passes run."""

    sink: list | None = None

    def __call__(self, graph: GraphLowering) -> None:
        super().__call__(graph)
        if _CapturePasses.sink is not None:
            _CapturePasses.sink.clear()
            _CapturePasses.sink.extend(graph.operations)


class TestMatmulNPadding(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0xAFFE)
        self.captured: list = []
        _CapturePasses.sink = self.captured
        self.patchers: list[Any] = [
            t_inductor_config.patch("force_disable_caches", True),
            ts_inductor_config.patch("sencores", 32),
            patch.object(passes, "CustomPreSchedulingPasses", _CapturePasses),
        ]
        for p in self.patchers:
            p.__enter__()
        torch.compiler.reset()

    def tearDown(self) -> None:
        _CapturePasses.sink = None
        for p in self.patchers:
            p.__exit__(None, None, None)
        torch.compiler.reset()

    def _matmul_ops(self) -> list[ComputedBuffer]:
        return [
            op
            for op in self.captured
            if isinstance(op, ComputedBuffer)
            and isinstance(op.data, Reduction)
            and op.data.reduction_type == BATCH_MATMUL_OP
        ]

    def _run(self, vocab: int):
        x_cpu = torch.randn(1, HIDDEN, dtype=torch.float16)
        w_cpu = torch.randn(vocab, HIDDEN, dtype=torch.float16)
        x = x_cpu.to("spyre")
        w = w_cpu.to("spyre")
        compiled = torch.compile(lambda a, b: F.linear(a, b), fullgraph=True)
        out = compiled(x, w)
        return out, x_cpu, w_cpu

    def test_prime_stick_vocab_pads_and_matches(self) -> None:
        """vocab 49216 (769 sticks, prime) -> N padded to 49280 (770), output
        narrowed back to 49216 and numerically correct."""
        vocab = 49216
        out, x_cpu, w_cpu = self._run(vocab)

        # Output narrowed back to the true vocab size.
        self.assertEqual(tuple(out.shape), (1, vocab))

        # The matmul itself was widened to a composite stick count (770*64).
        mms = self._matmul_ops()
        self.assertTrue(mms, "expected a BATCH_MATMUL_OP in the graph")
        padded_n = int(mms[0].get_size()[-1])
        self.assertEqual(padded_n, 49280, f"matmul N should be padded to 49280, got {padded_n}")

        # Numerics match CPU on the true columns.
        ref = F.linear(x_cpu, w_cpu)
        torch.testing.assert_close(out.to("cpu"), ref, rtol=1e-2, atol=1e-2)

    def test_composite_vocab_no_pad(self) -> None:
        """vocab 49152 (768 sticks, composite) -> span_reduction can split it, so
        no N padding is inserted."""
        vocab = 49152
        out, _, _ = self._run(vocab)
        self.assertEqual(tuple(out.shape), (1, vocab))
        padded_n = int(self._matmul_ops()[0].get_size()[-1])
        self.assertEqual(padded_n, vocab, "composite vocab must not be padded")

    def test_small_matmul_no_pad(self) -> None:
        """Small N (64 sticks) is under the span limit -> no padding."""
        vocab = HIDDEN  # 4096 -> 64 sticks
        out, _, _ = self._run(vocab)
        self.assertEqual(tuple(out.shape), (1, vocab))
        padded_n = int(self._matmul_ops()[0].get_size()[-1])
        self.assertEqual(padded_n, vocab, "small matmul must not be padded")


if __name__ == "__main__":
    unittest.main()
