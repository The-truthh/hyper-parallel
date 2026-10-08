# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""NPU full-head DSA parity, including real CP and FSDP gradient averaging."""

import copy
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch_npu

from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.components.functional.dsa_indexer import dsa_indexer
from hyper_parallel.components.functional.dsa_kl_loss import dsa_kl_loss
from hyper_parallel.components.functional.dsa_sparse_attention import dsa_sparse_attention
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.distributed.context_parallel.attention import _dsa_cp_alltoall


def _indexer(module, query, key, weights, q_lengths, k_lengths):
    """Expose the same functional boundary used by the model adapter."""
    return dsa_indexer(query, key, weights, q_lengths, k_lengths, module.index_topk)


def _sparse(module, query, key, value, attention_mask, **kwargs):
    """Keep the production sparse attention kernel inside the CP wrapper."""
    del module, value, attention_mask
    return dsa_sparse_attention(query, key, kwargs["q_pe"], kwargs["k_pe"],
                                kwargs["topk_indices"], kwargs["scale"],
                                kwargs["actual_q_len"], kwargs["actual_kv_len"])


class _NativeKL:
    """Adapter contract for the production scalar KL autograd bridge."""

    apply = staticmethod(dsa_kl_loss)


class _DSABanks(nn.Module):
    """Trainable index tensors plus a head weight shared across all tokens."""

    def __init__(self) -> None:
        """Use a divisible token axis for FSDP's ordinary shard contract."""
        super().__init__()
        self.query_bank = nn.Parameter(torch.randn(4096, 24, 128) * 0.1)
        self.key_bank = nn.Parameter(torch.randn(4096, 1, 128) * 0.1)
        self.merge_weights = nn.Parameter(torch.randn(24) * 0.1)
        self.num_heads = 48
        self.index_topk = 2048
        self.freeze_dsa = False
        self.rank = 0
        self.cp_size = 1
        self.symbols = SimpleNamespace(dsa_lightning_indexer_forward=_indexer,
                                       SparseLightningIndexerKLLossTrainFunction=_NativeKL)
        self.functions = {"dsa_sparse_attention": _sparse}

    def _attention_and_loss(self, query, key, query_rope, key_rope, enabled):
        """Run index selection, sparse teacher, then the complete KL boundary."""
        part = slice(self.rank * 4096 // self.cp_size, (self.rank + 1) * 4096 // self.cp_size)
        iq = self.query_bank[part].unsqueeze(0).to(torch.bfloat16)
        ik = self.key_bank[part].unsqueeze(0).to(torch.bfloat16)
        weights = self.merge_weights.to(torch.bfloat16).view(1, 1, 24).expand(1, iq.size(1), 24)
        lengths = torch.tensor([3072, 4096], device=query.device, dtype=torch.int32)
        indices, iq_flat, ik_flat, weights_flat = self.symbols.dsa_lightning_indexer_forward(
            self, iq, ik, weights, lengths, lengths)
        self.freeze_dsa = not enabled
        _, maximum, total = self.functions["dsa_sparse_attention"](
            self, query[:, part], key[:, part], key[:, part], None,
            q_pe=query_rope[:, part], k_pe=key_rope[:, part], topk_indices=indices,
            scale=192 ** -0.5, actual_q_len=lengths, actual_kv_len=lengths)
        if not enabled:
            return None, indices
        loss = self.symbols.SparseLightningIndexerKLLossTrainFunction.apply(
            iq_flat, ik_flat, weights_flat, query[:, part].flatten(0, 1), key[:, part].flatten(0, 1),
            indices, maximum, total, query_rope[:, part].flatten(0, 1), key_rope[:, part].flatten(0, 1),
            lengths, lengths, 192 ** -0.5, 0.1)
        return loss, indices

    def _compute(self, query, key, query_rope, key_rope, scenario):
        """Exercise serial layers and a disabled auxiliary producer."""
        if scenario == "frozen_then_active":
            self._attention_and_loss(query, key, query_rope, key_rope, False)
        loss, indices = self._attention_and_loss(query, key, query_rope, key_rope, True)
        if scenario == "two_active":
            second, _ = self._attention_and_loss(-query, key, -query_rope, key_rope, True)
            loss = loss + second
        return loss, indices

    def forward(
            self, query: torch.Tensor, key: torch.Tensor, query_rope: torch.Tensor,
            key_rope: torch.Tensor, scenario: str, recompute: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Optionally replay the complete sequence of model-owned boundaries."""
        if recompute:
            return checkpoint(self._compute, query, key, query_rope, key_rope, scenario, use_reentrant=False)
        return self._compute(query, key, query_rope, key_rope, scenario)


def _relative_l2(actual, expected):
    """Measure a whole logical tensor, not a shard-dependent norm."""
    return float(((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)).item())


def test_dsa_cp_fsdp_full_heads():
    """CP2/4 preserve selection, full teacher, FSDP gradients and SGD updates."""
    rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(rank)
    dist.init_process_group("hccl")
    try:
        world = dist.get_world_size()
        mesh = init_device_mesh("npu", (world,), mesh_dim_names=("cp",))
        for recompute in (False, True):
            for scenario in ("active_only", "frozen_then_active", "two_active"):
                torch.manual_seed(88)
                reference = _DSABanks().npu()
                candidate = copy.deepcopy(reference)
                candidate.rank, candidate.cp_size = rank, world
                context = SimpleNamespace(group=mesh.get_group(), rank=rank, size=world)
                _dsa_cp_alltoall(candidate.symbols, candidate.functions, context)
                fully_shard(candidate, mesh=mesh, mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False))
                reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
                candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01)
                teacher = [(torch.randn(1, 4096, heads, dim, device="npu", dtype=torch.bfloat16) * 0.1)
                           for heads, dim in ((48, 512), (1, 512), (48, 64), (1, 64))]
                for step in range(2):
                    reference_optimizer.zero_grad(set_to_none=True)
                    candidate_optimizer.zero_grad(set_to_none=True)
                    expected_loss, expected_indices = reference(*teacher, scenario, recompute)
                    actual_loss, actual_indices = candidate(*teacher, scenario, recompute)
                    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
                    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-8)
                    expected_loss.backward()
                    actual_loss.backward()
                    gradient_errors = {}
                    for (name, actual), (_, expected) in zip(
                            candidate.named_parameters(), reference.named_parameters()):
                        gradient = actual.grad.full_tensor()
                        assert bool(torch.isfinite(gradient).all().item()), f"Non-finite gradient: {name}"
                        error = _relative_l2(gradient, expected.grad)
                        gradient_errors[name] = error
                        assert error < 0.01, f"Gradient {name}: relative_l2={error}, limit=0.01"
                    reference_optimizer.step()
                    candidate_optimizer.step()
                    for (name, actual), (_, expected) in zip(
                            candidate.named_parameters(), reference.named_parameters()):
                        value = actual.full_tensor()
                        torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-6,
                                                   msg=f"Post-update parameter: {name}")
                    if rank == 0:
                        print(json.dumps({"cp": world, "recompute": recompute, "scenario": scenario,
                                          "step": step + 1, "loss": float(actual_loss.item()),
                                          "gradient_relative_l2": gradient_errors}), flush=True)
    finally:
        dist.destroy_process_group()
