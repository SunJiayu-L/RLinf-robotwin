# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Check that policy filtering preserves the original action window indices."""

from types import SimpleNamespace

import pytest

from rlinf.data.datasets.recap.cfg_model import AdvantagePreservingDataset


def test_excluded_samples_never_reach_transform():
    class Windows:
        def __len__(self):
            return 404

        def __getitem__(self, index):
            assert index <= 350 or index >= 400
            return {"source_index": index}

    hf = [{"episode_index": 0, "frame_index": t} for t in range(400)]
    hf.extend({"episode_index": 1, "frame_index": t} for t in range(4))

    class Metadata:
        def __len__(self):
            return len(hf)

        def __getitem__(self, key):
            return [row[key] for row in hf]

    base = SimpleNamespace(hf_dataset=Metadata())
    labels = {
        (0, t): True if t < 300 else False if t <= 350 else None for t in range(400)
    }
    labels.update({(1, t): True for t in range(4)})
    dataset = AdvantagePreservingDataset(base, Windows(), labels)
    assert len(dataset) == 355
    assert dataset[299] == {"source_index": 299, "advantage": True}
    assert dataset[300] == {"source_index": 300, "advantage": False}
    assert dataset[350] == {"source_index": 350, "advantage": False}
    assert dataset[351] == {"source_index": 400, "advantage": True}

    del labels[0, 351]
    with pytest.raises(ValueError, match="not found in advantages lookup"):
        AdvantagePreservingDataset(base, Windows(), labels)
