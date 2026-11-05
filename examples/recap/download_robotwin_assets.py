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

"""Download the official RECAP critic backbones without changing packages."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path("/data2/sjy/recap_assets")
root.mkdir(parents=True, exist_ok=True)
results = {}
for repo in ["google/gemma-3-270m", "google/siglip2-so400m-patch14-224"]:
    target = root / repo.split("/")[-1]
    try:
        snapshot_download(
            repo,
            local_dir=target,
            max_workers=4,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "tokenizer*",
                "vocab*",
            ],
        )
        results[repo] = {"status": "READY", "path": str(target)}
    except Exception as error:
        results[repo] = {"status": "FAILED", "error": str(error)}
    (root / "download_status.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({repo: results[repo]}), flush=True)
