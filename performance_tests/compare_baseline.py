#!/usr/bin/env python3
"""Compare a JSON result of poisson_cube_multigrid with a stored baseline.

  compare_baseline.py result.json baseline.json [--tolerance 0.15] [--update]

Cycles are matched by their number of DoFs. The CG iteration count has to be
the same; a timing fails if it is more than the tolerance slower than the
baseline, and is reported if it is more than the tolerance faster. Without a
baseline file nothing is compared (exit 0); --update copies the result to the
baseline.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

TIMINGS = ["mv_outer", "mv_inner", "cg_time", "setup_time"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("baseline")
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    result = json.loads(Path(args.result).read_text())
    baseline_path = Path(args.baseline)

    if args.update:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.result, baseline_path)
        print(f"baseline updated: {baseline_path}")
        return 0

    if not baseline_path.exists():
        print(f"no baseline {baseline_path}; record one with --update")
        return 0

    baseline = json.loads(baseline_path.read_text())
    for key in ["variant", "dim", "degree", "n_ranks", "n_pre_smooth", "n_post_smooth"]:
        if result.get(key) != baseline.get(key):
            print(f"FAILED: {key} = {result.get(key)}, baseline {baseline.get(key)}")
            return 1

    reference = {c["dofs"]: c for c in baseline["cycles"]}
    failed = False
    print(f"{'dofs':>10} {'metric':>10} {'result':>11} {'baseline':>11} {'ratio':>6}")
    for cycle in result["cycles"]:
        ref = reference.get(cycle["dofs"])
        if ref is None:
            print(f"{cycle['dofs']:>10} not in the baseline")
            continue
        if cycle["cg_its"] != ref["cg_its"]:
            print(f"{cycle['dofs']:>10} {'cg_its':>10} {cycle['cg_its']:>11} {ref['cg_its']:>11}  FAILED")
            failed = True
        for metric in TIMINGS:
            ratio = cycle[metric] / ref[metric]
            note = ""
            if ratio > 1 + args.tolerance:
                note = "  FAILED (slower)"
                failed = True
            elif ratio < 1 - args.tolerance:
                note = "  faster"
            print(f"{cycle['dofs']:>10} {metric:>10} {cycle[metric]:>11.3e} {ref[metric]:>11.3e} {ratio:>6.2f}{note}")

    print("FAILED" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
