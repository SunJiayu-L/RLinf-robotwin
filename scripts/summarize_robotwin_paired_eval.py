#!/usr/bin/env python3
"""Summarize RLinf JSONL results against a paired RoboTwin manifest."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("result_paths", type=Path, nargs="+")
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest_path.read_text())
    episodes = manifest[args.task]["episodes"]
    expected = {episode["seed"]: episode for episode in episodes}

    actual = {}
    for result_path in args.result_paths:
        for line in result_path.read_text().splitlines():
            record = json.loads(line)
            seed = record["seed"]
            if seed in actual:
                raise ValueError(f"Duplicate result for seed {seed}")
            actual[seed] = record

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"Result set mismatch: missing={missing}, unexpected={unexpected}"
        )

    instruction_mismatches = [
        seed
        for seed, episode in expected.items()
        if actual[seed]["instruction"] != episode["instruction"]
    ]
    if instruction_mismatches:
        raise ValueError(
            f"Instruction mismatch for seeds: {instruction_mismatches}"
        )

    reference_successes = sum(
        bool(episode["reference_success"]) for episode in episodes
    )
    rlinf_successes = sum(
        bool(actual[episode["seed"]]["success"]) for episode in episodes
    )
    total = len(episodes)
    reference_rate = reference_successes / total
    rlinf_rate = rlinf_successes / total

    disagreements = [
        {
            "seed": episode["seed"],
            "reference_success": episode["reference_success"],
            "rlinf_success": actual[episode["seed"]]["success"],
            "rlinf_success_step": actual[episode["seed"]]["success_step"],
            "instruction": episode["instruction"],
        }
        for episode in episodes
        if bool(episode["reference_success"])
        != bool(actual[episode["seed"]]["success"])
    ]

    report = {
        "task": args.task,
        "episodes": total,
        "reference_successes": reference_successes,
        "reference_success_rate": reference_rate,
        "rlinf_successes": rlinf_successes,
        "rlinf_success_rate": rlinf_rate,
        "absolute_error": abs(rlinf_rate - reference_rate),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
