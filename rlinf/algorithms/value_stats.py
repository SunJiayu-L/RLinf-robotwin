# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pool critic diagnostics across valid samples instead of microbatch variances."""

import torch


def value_sufficient_statistics(values, returns, mask=None):
    """Return additive FP64 statistics; callers sum them across batches/ranks."""
    values, returns = torch.broadcast_tensors(values.detach(), returns.detach())
    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    else:
        mask = torch.broadcast_to(mask, values.shape)
    y = returns[mask].double()
    v = values[mask].double()
    residual = y - v
    return torch.stack(
        [
            y.new_tensor(y.numel()),
            y.sum(),
            y.square().sum(),
            v.sum(),
            v.square().sum(),
            residual.sum(),
            residual.square().sum(),
            residual.abs().sum(),
        ]
    )


def value_metrics_from_statistics(stats):
    """Compute pooled diagnostics, leaving EV undefined for constant returns."""
    count, sy, sy2, sv, sv2, sr, sr2, sa = stats.detach().double()
    denominator = count.clamp_min(1)
    mean_return = sy / denominator
    mean_value = sv / denominator
    var_return = (sy2 / denominator - mean_return.square()).clamp_min(0)
    var_value = (sv2 / denominator - mean_value.square()).clamp_min(0)
    var_residual = (sr2 / denominator - (sr / denominator).square()).clamp_min(0)
    defined = (count > 1) & (var_return > 1e-12)
    ev = torch.where(
        defined,
        1 - var_residual / var_return.clamp_min(1e-12),
        stats.new_tensor(float("nan")),
    )
    return {
        "critic/explained_variance": ev,
        "critic/ev_defined": defined.double(),
        "critic/valid_samples": count,
        "critic/return_mean": mean_return,
        "critic/return_std": var_return.sqrt(),
        "critic/value_mean": mean_value,
        "critic/value_std": var_value.sqrt(),
        "critic/mse": sr2 / denominator,
        "critic/mae": sa / denominator,
    }
