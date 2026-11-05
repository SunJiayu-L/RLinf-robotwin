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

"""Independent experiments must not overwrite configs or share scalar steps."""

from pathlib import Path

from omegaconf import OmegaConf

from rlinf.utils import metric_logger


def test_two_experiments_have_distinct_tensorboard_runs(tmp_path, monkeypatch):
    paths = []

    class WriterStub:
        def __init__(self, path):
            paths.append(Path(path))

        def finish(self):
            pass

    monkeypatch.setattr(metric_logger, "_TensorboardLogger", WriterStub)
    for name in ["ppo100", "ppo200"]:
        config = OmegaConf.create(
            {
                "runner": {
                    "logger": {
                        "log_path": str(tmp_path),
                        "experiment_name": name,
                        "logger_backends": ["tensorboard"],
                    }
                }
            }
        )
        metric_logger.MetricLogger(config)
    assert paths == [
        tmp_path / "ppo100" / "tensorboard",
        tmp_path / "ppo200" / "tensorboard",
    ]
    for name, path in zip(["ppo100", "ppo200"], paths, strict=True):
        saved = OmegaConf.load(path / "config.yaml")
        assert saved.runner.logger.experiment_name == name
    assert not (tmp_path / "tensorboard").exists()
