#!/usr/bin/env python3
"""Tabulate the Nsight Compute metrics of the BK3 cell kernels side by side.

Usage: python3 bk3_kernel_profile_summary.py <directory>

The directory holds <kernel>_<number>.csv files written by
bk3_kernel_profile_jupiter.sbatch (`ncu --import <rep> --csv --page raw`, one
row per profiled launch). For each number type the median over the launches
of every kernel is printed, with the ratio to the plain kernel. If a report
holds launches of several kernels or sizes, the group with the longest
median duration (the fine level) is taken.
"""

import csv
import statistics
import sys
from pathlib import Path

# metric names of this Nsight Compute version, shared with the transfer kernels
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "performance_tests" / "mg_transfer_comparison"))
from profile_summary import METRICS, to_float  # noqa: E402

KERNELS = ["abstracted", "plain", "plain_constant"]
DURATION = "gpu__time_duration.sum"
TIME_SCALE = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3}


def read_fine_level(path):
    """Columns, units and the rows of the longest-running kernel group."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        rows = list(reader)
    if not header or "Kernel Name" not in header:
        return None
    col = {name: i for i, name in enumerate(header)}
    units = rows[0] if rows and to_float(rows[0][col[DURATION]]) is None else None
    data = rows[1:] if units else rows
    if not data:
        return None

    groups = {}
    for r in data:
        key = (r[col["Kernel Name"]], r[col.get("launch__grid_size", 0)], r[col.get("launch__block_size", 0)])
        groups.setdefault(key, []).append(r)

    def median_duration(rs):
        xs = [x for x in (to_float(r[col[DURATION]]) for r in rs) if x is not None]
        return statistics.median(xs) if xs else 0.0

    key = max(groups, key=lambda k: median_duration(groups[k]))
    return col, units, groups[key], key[0], len(groups)


def median_of(col, rows, metric):
    if metric not in col:
        return None
    xs = [x for x in (to_float(r[col[metric]]) for r in rows) if x is not None]
    return statistics.median(xs) if xs else None


def main(directory):
    directory = Path(directory)
    numbers = sorted({p.stem.rsplit("_", 1)[1] for p in directory.glob("*_*.csv")})
    if not numbers:
        sys.exit(f"no <kernel>_<number>.csv files in {directory}")

    for number in numbers:
        reports = {}
        for kernel in KERNELS:
            path = directory / f"{kernel}_{number}.csv"
            if path.exists():
                report = read_fine_level(path)
                if report:
                    reports[kernel] = report
        if not reports:
            continue

        print(f"\n== {number} ==")
        for kernel, (_, _, rows, name, n_groups) in reports.items():
            more = f", {n_groups} kernel groups in the report" if n_groups > 1 else ""
            print(f"{kernel}: {len(rows)} launches{more}\n  {name[:300]}")

        kernels = list(reports)
        print(f"\n{'metric':<32}" + "".join(f"{k:>18}" for k in kernels) + "".join(f"{k + '/plain':>22}" for k in kernels if k != "plain" and "plain" in reports))
        for metric, label in METRICS:
            vals = {}
            for kernel, (col, units, rows, _, _) in reports.items():
                v = median_of(col, rows, metric)
                if v is not None and metric == DURATION and units:
                    v *= TIME_SCALE.get(units[col[metric]], 1.0)
                vals[kernel] = v
            if all(v is None for v in vals.values()):
                continue
            line = f"{label:<32}" + "".join(f"{vals[k]:>18,.1f}" if vals[k] is not None else f"{'-':>18}" for k in kernels)
            ref = vals.get("plain")
            for k in kernels:
                if k == "plain" or "plain" not in reports:
                    continue
                line += f"{vals[k] / ref:>22.3f}" if vals[k] is not None and ref else f"{'-':>22}"
            print(line)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
