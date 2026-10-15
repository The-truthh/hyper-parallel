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
"""AutoModels-private vocab-parallel cross entropy (local-first).

Merges the former ``components/loss/loss_parallel_ops.py`` kernel with the
``loss_parallel_ops_common.py`` validation helpers into a single private
production implementation. The leading underscore marks this module as
internal: the generic DTensor loss API stays in ``core.tensor_parallel``
(frozen), while AutoModels production loss uses this module.

The public-in-package entry is :func:`vocab_parallel_cross_entropy_local`:
it takes the *local* logits shard plus the global vocabulary size, the mesh
and the mesh axis that shards the class dimension — no temporary DTensor is
created per call. The 2D mesh-axis recognition and the "class dimension
sharded by exactly one mesh axis" validation from the former AutoModels
DTensor wrapper are absorbed by :func:`_resolve_class_mesh_dim`.
"""

from __future__ import annotations

__all__ = [
    "vocab_parallel_cross_entropy_local",
    "distributed_log_softmax",
    "distributed_nll_loss_forward",
    "DistributedCrossEntropyFunction",
]

from typing import Any, NamedTuple, Optional, Tuple, TYPE_CHECKING

import torch
import torch.distributed.nn.functional as dist_func
from torch import Tensor

if TYPE_CHECKING:
    from hyper_parallel.core.dtensor.device_mesh import DeviceMesh

_REDUCE_OP_MAP = {
    "sum": torch.distributed.ReduceOp.SUM,
    "prod": torch.distributed.ReduceOp.PRODUCT,
    "max": torch.distributed.ReduceOp.MAX,
    "min": torch.distributed.ReduceOp.MIN,
    "all": torch.distributed.ReduceOp.MIN,
    "avg": torch.distributed.ReduceOp.SUM,
}


def _differentiable_all_reduce(data: Tensor, op: str, group: Any) -> Tensor:
    """Autograd-aware all-reduce: backward is an identity, the loss is scaled by world size."""
    if not data.is_contiguous() or data.storage_offset() != 0:
        data = data.contiguous()
    reduce_op = _REDUCE_OP_MAP.get(op, torch.distributed.ReduceOp.SUM)
    return dist_func.all_reduce(data, op=reduce_op, group=group)


def _eager_all_reduce(data: Tensor, op: str, group: Any) -> Tensor:
    """All-reduce disposable forward intermediates inside custom autograd."""
    if not data.is_contiguous() or data.storage_offset() != 0:
        data = data.contiguous()
    reduce_op = _REDUCE_OP_MAP.get(op, torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(data, op=reduce_op, group=group)
    return data


def _validate_target_type_base(is_floating: bool) -> None:
    """Validate target type (base check).

    Args:
        is_floating: Whether target tensor has floating dtype.

    Raises:
        ValueError: If target is floating point (probabilistic).
    """
    if is_floating:
        raise ValueError(
            "Probabilistic target (float) is not supported in loss_parallel. "
            "Target must be class indices (int64)."
        )


def _resolve_class_mesh_dim(mesh: "DeviceMesh", mesh_dim: Optional[int]) -> int:
    """Identify the single mesh axis that shards the class dimension.

    Absorbs the 2D mesh-axis recognition of the former DTensor-based
    implementation: an explicit ``mesh_dim`` is validated against the mesh,
    a 1D mesh resolves to axis 0, and a multi-dimensional mesh resolves the
    axis named ``"tp"``. The class dimension must be sharded by exactly one
    mesh axis.

    Args:
        mesh: Mesh holding the class-sharding axis.
        mesh_dim: Explicitly declared class-sharding axis, or ``None`` to
            resolve it from the mesh shape and dimension names.

    Returns:
        The resolved mesh axis index.

    Raises:
        ValueError: If no unique class-sharding axis can be identified.
    """
    if mesh_dim is not None:
        if not isinstance(mesh_dim, int) or not 0 <= mesh_dim < mesh.ndim:
            raise ValueError(
                "Expected the class dimension to be sharded on exactly one "
                f"mesh dimension. Got mesh_dim={mesh_dim} for a "
                f"{mesh.ndim}D mesh."
            )
        return mesh_dim
    if mesh.ndim == 1:
        return 0
    names = mesh.mesh_dim_names or ()
    matches = [axis for axis, name in enumerate(names) if name == "tp"]
    if len(matches) != 1:
        raise ValueError(
            "Expected the class dimension to be sharded on exactly one mesh "
            f"dimension. Got {mesh.ndim}D mesh with dimension names {names}; "
            "pass mesh_dim explicitly or name the sharding axis 'tp'."
        )
    return matches[0]


def _is_floating_torch(tensor: Tensor) -> bool:
    """Check if PyTorch tensor is floating point."""
    return tensor.is_floating_point()


def _compute_vocab_start(vocab_size: int, tp_size: int, rank: int) -> int:
    """Compute the starting index for this rank's vocab shard.

    Args:
        vocab_size: Total vocabulary size.
        tp_size: Tensor parallel world size.
        rank: Current rank in TP mesh.

    Returns:
        Starting index of this rank's vocab shard.

    Note:
        This follows torch.chunk behavior: chunk_size = ceil(vocab_size/tp_size),
        and each rank's start = rank * chunk_size. The last rank may have fewer elements.
    """
    chunk_size = (vocab_size + tp_size - 1) // tp_size  # ceil division
    return rank * chunk_size


def distributed_log_softmax(
    logits_local: Tensor,
    dim: int,
    mesh: DeviceMesh,
    mesh_dim: int = 0,
) -> Tensor:
    """K1: Stable log-softmax on class-sharded dimension.

    Args:
        logits_local: Local logits shard.
        dim: Class dimension.
        mesh: DeviceMesh.
        mesh_dim: Mesh dimension (default 0).

    Returns:
        Local log-softmax with unchanged layout.

    Communication:
        MAX + SUM all_reduce
    """
    max_local = logits_local.max(dim=dim, keepdim=True).values

    group = mesh.get_group(mesh_dim)
    max_global = _differentiable_all_reduce(max_local, op="max", group=group)

    exp_local = (logits_local - max_global).exp()

    sum_local = exp_local.sum(dim=dim, keepdim=True)

    sum_global = _differentiable_all_reduce(sum_local, op="sum", group=group)

    log_softmax = logits_local - max_global - sum_global.log()

    return log_softmax


def distributed_nll_loss_forward(
    log_probs: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    ignore_index: int,
    reduction: str,
    vocab_start: int,
    vocab_end: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """K2: Index target + optional weight + reduction.

    Args:
        log_probs: Sharded log_probs.
        target: Target class indices.
        weight: Optional weights.
        ignore_index: Index to ignore.
        reduction: Reduction method.
        vocab_start: Start index of this vocab shard.
        vocab_end: End index of this vocab shard.

    Returns:
        Tuple of (loss, total_weight, target_mask, vocab_start_tensor).
    """
    target_flat = target.flatten()
    valid_mask = target_flat != ignore_index
    target_mask = (
        (target_flat >= vocab_start)
        & (target_flat < vocab_end)
        & valid_mask
    )
    local_target = torch.where(
        target_mask,
        target_flat - vocab_start,
        torch.zeros_like(target_flat),
    )
    row_indices = torch.arange(target_flat.numel(), device=target.device, dtype=torch.long)
    selected_log_probs = log_probs.reshape(-1, log_probs.shape[-1])[row_indices, local_target]

    if weight is not None:
        safe_target = torch.where(valid_mask, target_flat, torch.zeros_like(target_flat))
        local_weights = weight[safe_target] * target_mask.to(weight.dtype)
    else:
        local_weights = target_mask.to(log_probs.dtype)
    local_weights = local_weights.to(log_probs.dtype)
    loss_per_token = -selected_log_probs * local_weights
    total_weight = local_weights.sum().reshape(1)

    if reduction == "none":
        loss = loss_per_token.reshape(target.shape)
    else:
        loss = loss_per_token.sum().reshape(1)

    return loss, total_weight, target_mask, torch.tensor(
        vocab_start, dtype=torch.long, device=log_probs.device
    )


class _CrossEntropyGradients(NamedTuple):
    """Gradients aligned with the eight inputs of the autograd function."""

    input_local: Tensor
    target: None = None
    weight: None = None
    ignore_index: None = None
    reduction: None = None
    vocab_size: None = None
    mesh: None = None
    mesh_dim: None = None


class _CrossEntropyContext(NamedTuple):
    """State required by the custom cross-entropy backward pass."""

    softmax_local: Tensor
    masked_target: Tensor
    local_target_mask: Tensor
    token_weights: Tensor
    total_weight: Tensor
    reduction: str


def _compute_cross_entropy_gradient(
    state: _CrossEntropyContext,
    grad_output: Tensor,
) -> Tensor:
    """Compute the local logits gradient for one reduction mode."""
    if state.reduction == "mean":
        grad_scale = grad_output / state.total_weight.clamp(min=1e-12)
    elif state.reduction == "sum":
        grad_scale = grad_output
    else:
        grad_scale = grad_output.flatten()

    row_indices = torch.arange(
        state.masked_target.numel(),
        device=state.masked_target.device,
        dtype=torch.long,
    )
    grad_input = state.softmax_local
    grad_input[row_indices, state.masked_target] -= state.local_target_mask.to(grad_input.dtype)
    row_scale = state.token_weights * grad_scale
    grad_input.mul_(row_scale.unsqueeze(-1))
    return grad_input


class DistributedCrossEntropyFunction(torch.autograd.Function):
    """K3: Fused backward for distributed cross_entropy."""

    @staticmethod
    def forward(
        ctx: Any,
        input_local: Tensor,
        target: Tensor,
        weight: Optional[Tensor],
        ignore_index: int,
        reduction: str,
        vocab_size: int,
        mesh: DeviceMesh,
        mesh_dim: int,
    ) -> Tensor:
        """Compute distributed cross entropy and save fixed-shape backward state.

        Args:
            ctx: Autograd context used to save tensors for backward.
            input_local: Logits for the vocabulary shard owned by this rank.
            target: Global target class indices.
            weight: Optional per-class loss weights.
            ignore_index: Target value excluded from the loss.
            reduction: Reduction method: ``none``, ``mean``, or ``sum``.
            vocab_size: Global vocabulary size across all shard ranks.
            mesh: Device mesh containing the vocabulary-sharding axis.
            mesh_dim: Mesh dimension that shards the vocabulary.

        Returns:
            The globally reduced cross-entropy loss.
        """
        ctx.vocab_start = _compute_vocab_start(
            vocab_size, mesh.size(mesh_dim), mesh.get_local_rank(mesh_dim)
        )
        ctx.vocab_end = ctx.vocab_start + input_local.shape[-1]
        ctx.reduction = reduction
        group = mesh.get_group(mesh_dim)
        target_flat = target.flatten()
        valid_mask = target_flat != ignore_index
        local_target_mask = (
            (target_flat >= ctx.vocab_start)
            & (target_flat < ctx.vocab_end)
            & valid_mask
        )
        masked_target = torch.where(
            local_target_mask,
            target_flat - ctx.vocab_start,
            torch.zeros_like(target_flat),
        )
        row_indices = torch.arange(target_flat.numel(), device=target.device, dtype=torch.long)

        logits_max = input_local.max(dim=-1, keepdim=True).values
        logits_max = _eager_all_reduce(logits_max, op="max", group=group)
        softmax_local = input_local - logits_max
        predicted_logits = softmax_local[row_indices, masked_target].clone()
        predicted_logits.mul_(local_target_mask.to(predicted_logits.dtype))
        predicted_logits = _eager_all_reduce(predicted_logits, op="sum", group=group)

        softmax_local.exp_()
        sum_exp_logits = softmax_local.sum(dim=-1, keepdim=True)
        sum_exp_logits = _eager_all_reduce(sum_exp_logits, op="sum", group=group)
        softmax_local.div_(sum_exp_logits)

        loss_per_token = sum_exp_logits.log().squeeze(-1) - predicted_logits
        if weight is not None:
            safe_target = torch.where(valid_mask, target_flat, torch.zeros_like(target_flat))
            token_weights = weight[safe_target] * valid_mask.to(weight.dtype)
        else:
            token_weights = valid_mask.to(input_local.dtype)
        token_weights = token_weights.to(input_local.dtype)
        loss_per_token.mul_(token_weights)
        total_weight = token_weights.sum().reshape(1)

        if reduction == "none":
            loss = loss_per_token.reshape(target.shape)
        else:
            loss = loss_per_token.sum().reshape(1)
            if reduction == "mean":
                loss = loss / total_weight

        ctx.save_for_backward(
            softmax_local,
            masked_target,
            local_target_mask,
            token_weights,
            total_weight,
        )
        return loss

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """Compute the local logit gradient without dynamic indexing.

        Args:
            ctx: Autograd context populated by :meth:`forward`.
            grad_output: Upstream gradient for the loss.

        Returns:
            Gradients for all forward arguments; only local logits receive one.
        """
        state = _CrossEntropyContext(
            *ctx.saved_tensors,
            reduction=ctx.reduction,
        )
        return _CrossEntropyGradients(_compute_cross_entropy_gradient(state, grad_output))


def vocab_parallel_cross_entropy_local(
    local_logits: Tensor,
    target: Tensor,
    *,
    vocab_size: int,
    mesh: "DeviceMesh",
    mesh_dim: Optional[int] = None,
    ignore_index: int = -100,
    reduction: str = "sum",
) -> Tensor:
    """Vocab-parallel cross entropy on a local logits shard (local-first).

    AutoModels production entry: the caller owns the sharding contract and
    passes the local shard directly, so no temporary DTensor is created per
    call. ``mesh``/``mesh_dim`` identify the single mesh axis that shards the
    class dimension; see :func:`_resolve_class_mesh_dim`.

    Args:
        local_logits: Local class-sharded logits, ``[..., local_vocab_size]``.
            Rank above 2 is flattened to ``[-1, local_vocab_size]`` together
            with ``target``.
        target: Class indices aligned with the leading dims of
            ``local_logits``.
        vocab_size: Global vocabulary size (across all shard ranks).
        mesh: Mesh holding the class-sharding axis.
        mesh_dim: Explicit class-sharding mesh axis; ``None`` resolves it
            from the mesh (1D mesh → axis 0; otherwise the axis named "tp").
        ignore_index: Target value excluded from the loss.
        reduction: Reduction method: 'none', 'mean', or 'sum'.

    Returns:
        Loss tensor.

    Raises:
        ValueError: If the class-sharding mesh axis cannot be uniquely
            identified, the target is probabilistic, or the reduction is
            invalid.
    """
    mesh_dim = _resolve_class_mesh_dim(mesh, mesh_dim)
    _validate_target_type_base(_is_floating_torch(target))
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'none', 'mean', or 'sum'.")

    local_vocab_size = local_logits.shape[-1]
    if local_logits.ndim > 2:
        local_logits = local_logits.reshape(-1, local_vocab_size)
        target = target.reshape(-1)

    return DistributedCrossEntropyFunction.apply(
        local_logits,
        target,
        None,
        ignore_index,
        reduction,
        vocab_size,
        mesh,
        mesh_dim,
    )
