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

"""Isolate evaluation noise from training RNG and repeat it across checkpoints."""

import random
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def evaluation_rng(seed, rank, device):
    """Restore Python/NumPy/Torch state after a reproducibly seeded evaluation."""
    if seed is None:
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    device = torch.device(device)
    devices = []
    if device.type == "cuda":
        devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    try:
        with torch.random.fork_rng(devices=devices):
            effective_seed = int(seed) + int(rank) * 100003
            random.seed(effective_seed)
            np.random.seed(effective_seed % 2**32)
            # Avoid touching CUDA generators belonging to any other device.
            torch.random.default_generator.manual_seed(effective_seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(effective_seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
