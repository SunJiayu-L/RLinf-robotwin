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

"""Resolve actual policy weights before considering base safetensor links."""

from pathlib import Path

import torch


def resolve_pt_checkpoint(path: str) -> Path | None:
    """Prefer a direct fine-tuned checkpoint over any fallback base model."""
    root = Path(path)
    if root.is_file() and root.suffix == ".pt":
        return root
    for relative in [
        "full_weights.pt",
        "model_state_dict/full_weights.pt",
        "actor/model_state_dict/full_weights.pt",
    ]:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def load_pt_policy(model, path: Path, strict: bool = True) -> dict:
    """Load SFT weights and reject silent incomplete initialization."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Empty/invalid policy state dictionary: {path}")
    incompatible = model.load_state_dict(state, strict=strict)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "num_tensors": len(state),
    }
