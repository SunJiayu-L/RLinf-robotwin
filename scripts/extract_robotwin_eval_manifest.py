#!/usr/bin/env python3
"""Extract an ordered seed/instruction manifest from a RoboTwin eval log."""

import argparse
import json
import re
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def extract_episodes(log_path: Path, limit: int | None) -> list[dict[str, object]]:
    text = ANSI_ESCAPE.sub("", log_path.read_text(errors="replace"))
    result_matches = re.findall(
        r"Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)",
        text,
        flags=re.DOTALL,
    )
    seeds = [int(seed) for _, _, seed in result_matches]
    instructions = [
        instruction.strip()
        for instruction in re.findall(r"successfully set instruction:(.*)", text)
    ]
    if len(seeds) != len(instructions):
        raise ValueError(
            f"Found {len(seeds)} seeds but {len(instructions)} instructions"
        )

    episodes = []
    previous_successes = 0
    for instruction, (successes, episode_count, seed) in zip(
        instructions, result_matches, strict=True
    ):
        successes = int(successes)
        episode_count = int(episode_count)
        episodes.append(
            {
                "seed": int(seed),
                "instruction": instruction,
                "reference_success": successes > previous_successes,
                "reference_episode_index": episode_count - 1,
            }
        )
        previous_successes = successes
    return episodes if limit is None else episodes[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    episodes = extract_episodes(args.log_path, args.limit)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps({args.task: {"episodes": episodes}}, indent=2) + "\n"
    )
    print(f"Wrote {len(episodes)} episodes to {args.output_path}")


if __name__ == "__main__":
    main()
