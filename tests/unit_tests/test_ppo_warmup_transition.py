# Copyright 2025 The RLinf Authors.
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

"""The critic warmup boundary must not reset Adam or strand its scheduler."""

from types import SimpleNamespace

import torch

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager


def test_warmup_preserves_optimizer_and_enables_actor_at_boundary():
    model = torch.nn.Module()
    model.actor = torch.nn.Linear(1, 1, bias=False)
    model.value_head = torch.nn.Linear(1, 1, bias=False)
    actor = model.actor.weight
    critic = model.value_head.weight
    actor.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {"params": [actor], "lr": 0.01},
            {"params": [critic], "lr": 0.01},
        ]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    harness = SimpleNamespace(
        optimizer=optimizer,
        lr_scheduler=scheduler,
        model=model,
        optimizer_steps=0,
        critic_warmup_steps=2,
        store_requires_grad_param_name=["actor.weight", "value_head.weight"],
        grad_scaler=torch.amp.GradScaler("cpu", enabled=False),
        _strategy=SimpleNamespace(
            clip_grad_norm_=lambda model: torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
        ),
    )
    initial_actor = actor.detach().clone()
    for step in [1, 2]:
        optimizer.zero_grad(set_to_none=True)
        model.value_head(torch.ones(1, 1)).sum().backward()
        FSDPModelManager.optimizer_step(harness)
        assert harness.optimizer is optimizer
        assert harness.lr_scheduler is scheduler
        assert scheduler.optimizer is optimizer
        assert optimizer.state[critic]["step"].item() == step
        assert torch.equal(actor, initial_actor)
        assert actor.requires_grad is (step == 2)
    assert harness.critic_warmup_steps == 0
    assert harness.store_requires_grad_param_name == []
    optimizer.zero_grad(set_to_none=True)
    (
        model.actor(torch.ones(1, 1)) + model.value_head(torch.ones(1, 1))
    ).sum().backward()
    FSDPModelManager.optimizer_step(harness)
    assert optimizer.state[critic]["step"].item() == 3
    assert not torch.equal(actor, initial_actor)
