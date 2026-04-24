# Copyright 2025 The Torch-Spyre Authors.
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

"""
LayerNorm FP32 precision comparison test.

Compares LayerNorm output divergence across four configurations against a
CPU fp32 baseline:
  1. CPU fp32 (baseline)
  2. CPU fp16
  3. Spyre fp32  (TORCH_SPYRE_FP32_LAYERNORM=1)
  4. Spyre DL16  (default)

Input: random N(0, 1/sqrt(4096)) tensor, shape (256, 4096).
"""

import os
import unittest.mock

import torch
import torch.nn as nn


HIDDEN_DIM = 4096
BATCH_SIZE = 256
INPUT_STD = 1.0 / (HIDDEN_DIM**0.5)


def make_inputs():
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(BATCH_SIZE, HIDDEN_DIM, generator=gen) * INPUT_STD
    weight = torch.randn(HIDDEN_DIM, generator=gen) * 0.1
    bias = torch.randn(HIDDEN_DIM, generator=gen) * 0.01
    return x, weight, bias


def layernorm_fn(x, weight, bias):
    return torch.nn.functional.layer_norm(
        x, [x.shape[-1]], weight=weight, bias=bias
    )


def run_cpu_fp32(x, weight, bias):
    return layernorm_fn(x.float(), weight.float(), bias.float())


def run_cpu_fp16(x, weight, bias):
    out = layernorm_fn(x.half(), weight.half(), bias.half())
    return out.float()


def run_spyre_compiled(x_fp16, weight_fp16, bias_fp16, fp32_layernorm=False):
    import torch_spyre  # noqa: F401

    device = torch.device("spyre")
    x_dev = x_fp16.to(device)
    w_dev = weight_fp16.to(device)
    b_dev = bias_fp16.to(device)

    env_patch = (
        {"TORCH_SPYRE_FP32_LAYERNORM": "1"} if fp32_layernorm else {}
    )

    @torch.compile(backend="inductor")
    def compiled_fn(x, w, b):
        return layernorm_fn(x, w, b)

    with unittest.mock.patch.dict(os.environ, env_patch):
        torch._dynamo.reset()
        out = compiled_fn(x_dev, w_dev, b_dev)

    return out.cpu().float()


def divergence(result, baseline):
    diff = (result - baseline).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rmse": diff.pow(2).mean().sqrt().item(),
    }


def main():
    x, weight, bias = make_inputs()

    print(f"Input shape: ({BATCH_SIZE}, {HIDDEN_DIM})")
    print(f"Input std target: {INPUT_STD:.6f}, actual: {x.std().item():.6f}")
    print()

    baseline = run_cpu_fp32(x, weight, bias)
    print(f"[1] CPU fp32 baseline — norm: {baseline.norm().item():.6f}")
    print()

    cpu_fp16 = run_cpu_fp16(x, weight, bias)
    d = divergence(cpu_fp16, baseline)
    print(f"[2] CPU fp16 vs baseline:")
    print(f"    max_abs={d['max_abs']:.6e}  mean_abs={d['mean_abs']:.6e}  rmse={d['rmse']:.6e}")
    print()

    spyre_fp32 = run_spyre_compiled(x.half(), weight.half(), bias.half(), fp32_layernorm=True)
    d = divergence(spyre_fp32, baseline)
    print(f"[3] Spyre fp32 vs baseline:")
    print(f"    max_abs={d['max_abs']:.6e}  mean_abs={d['mean_abs']:.6e}  rmse={d['rmse']:.6e}")
    print()

    spyre_dl16 = run_spyre_compiled(x.half(), weight.half(), bias.half(), fp32_layernorm=False)
    d = divergence(spyre_dl16, baseline)
    print(f"[4] Spyre DL16 vs baseline:")
    print(f"    max_abs={d['max_abs']:.6e}  mean_abs={d['mean_abs']:.6e}  rmse={d['rmse']:.6e}")
    print()

    d_fp32_vs_dl16 = divergence(spyre_fp32, spyre_dl16)
    print(f"[*] Spyre fp32 vs Spyre DL16:")
    print(f"    max_abs={d_fp32_vs_dl16['max_abs']:.6e}  mean_abs={d_fp32_vs_dl16['mean_abs']:.6e}  rmse={d_fp32_vs_dl16['rmse']:.6e}")


if __name__ == "__main__":
    main()
