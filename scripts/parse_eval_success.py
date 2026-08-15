#!/usr/bin/env python3
"""Parse eval/success_once out of a 5-task eval run_embodiment.log and pivot
into a candidate x task summary table.

Each 5-task eval job (submitted via my_docs/submit_eval_5task_generic.sh or
similar) prints, for each task in sequence:

    === Evaluating eval_kl001_s300_<task> ===
    ...
    [INFO ... RLinf] {'eval/success_once': array(0.625, dtype=float32), ...}

This script splits a log on the "=== Evaluating ..." markers and extracts
the LAST eval/success_once value in each task section (defensive against a
task section printing more than one such line).

Usage:
    python scripts/parse_eval_success.py \\
        --run mode1-a03 rep1 /path/to/run_embodiment.log \\
        --run mode1-a03 rep2 /path/to/run_embodiment.log \\
        ...
        --output summary.md

Or point it at a directory of run dirs, each containing run_embodiment.log,
matching a naming convention <prefix>-<candidate>-rep<N>-...:
    python scripts/parse_eval_success.py --scan-dir /path/to/rlinf-robotwin-runs --prefix wm-
"""

from __future__ import annotations

import argparse
import pathlib
import re
import statistics as st
from collections import defaultdict

TASK_MARKER_RE = re.compile(r"=== Evaluating eval_\S*_(bbh|ca|cb|hb|mpa) ===")
SUCCESS_ONCE_RE = re.compile(r"'eval/success_once':\s*array\(([0-9eE.+-]+)")
TASKS = ["bbh", "ca", "cb", "hb", "mpa"]


def parse_log(log_path: pathlib.Path) -> dict[str, float]:
    """Return {task: success_once} for one eval run log."""
    text = log_path.read_text(errors="ignore")
    # Split into sections keyed by task, in the order markers appear.
    matches = list(TASK_MARKER_RE.finditer(text))
    if not matches:
        raise ValueError(f"No '=== Evaluating ===' markers found in {log_path}")
    result: dict[str, float] = {}
    for i, m in enumerate(matches):
        task = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end]
        values = SUCCESS_ONCE_RE.findall(section)
        if not values:
            print(f"  WARNING: no eval/success_once found for task={task} in {log_path}")
            continue
        # Defensive: take the last match in the section.
        result[task] = float(values[-1])
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("CANDIDATE", "REP", "LOG_PATH"),
        default=[],
        help="One eval run: candidate name, repeat label, path to run_embodiment.log. Repeatable.",
    )
    ap.add_argument("--output", type=str, default=None, help="Write markdown table to this path (also prints to stdout).")
    args = ap.parse_args()

    if not args.run:
        ap.error("provide at least one --run CANDIDATE REP LOG_PATH")

    # candidate -> task -> [values across reps]
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    raw_rows: list[tuple[str, str, str, float]] = []

    for candidate, rep, log_path_str in args.run:
        log_path = pathlib.Path(log_path_str)
        if not log_path.exists():
            print(f"  WARNING: log not found, skipping: {log_path}")
            continue
        per_task = parse_log(log_path)
        for task, val in per_task.items():
            data[candidate][task].append(val)
            raw_rows.append((candidate, rep, task, val))

    lines: list[str] = []
    lines.append("| candidate | " + " | ".join(TASKS) + " | avg | min |")
    lines.append("|---|" + "---|" * (len(TASKS) + 2))
    for candidate in sorted(data.keys()):
        cells = []
        means = []
        for task in TASKS:
            vals = data[candidate].get(task, [])
            if vals:
                mean = st.mean(vals)
                means.append(mean)
                if len(vals) > 1:
                    spread = max(vals) - min(vals)
                    cells.append(f"{mean * 100:.1f}% (n={len(vals)}, Δ{spread * 100:.1f}pp)")
                else:
                    cells.append(f"{mean * 100:.1f}% (n=1)")
            else:
                cells.append("—")
        avg = st.mean(means) * 100 if means else float("nan")
        mn = min(means) * 100 if means else float("nan")
        lines.append(f"| {candidate} | " + " | ".join(cells) + f" | {avg:.1f}% | {mn:.1f}% |")

    table = "\n".join(lines)
    print(table)

    print("\n--- raw per-repeat values ---")
    for candidate, rep, task, val in raw_rows:
        print(f"{candidate}\t{rep}\t{task}\t{val * 100:.2f}%")

    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table + "\n")
        print(f"\nWrote table to {out_path}")


if __name__ == "__main__":
    main()
