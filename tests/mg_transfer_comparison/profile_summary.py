#!/usr/bin/env python3
"""Tabulate Nsight Compute metrics of the transfer kernels side by side.

Usage: python3 profile_summary.py results/profile-<jobid>-ncu.csv

The CSV is the output of `ncu --import <rep> --csv --page raw` (one row per
profiled launch, one column per metric; the first data row holds the units).
Launches are grouped by kernel variant (deal.II, copy, BK1; the copy with one
cell per team is told apart by its grid size) and by operation (prolongation
or restriction), and the median over the launches of each group is printed.
Stall values are warps stalled for that reason per issue-active cycle.
"""

import csv
import statistics
import sys

METRICS = [
    # (metric name, label); names as in `ncu --page raw` of Nsight Compute 2025.3
    ("gpu__time_duration.sum", "time [us]"),
    ("launch__grid_size", "grid size"),
    ("launch__block_size", "block size"),
    ("launch__registers_per_thread", "registers/thread"),
    ("launch__shared_mem_per_block_dynamic", "dyn. shared/block [B]"),
    ("launch__occupancy_limit_registers", "occupancy limit (regs)"),
    ("launch__occupancy_limit_shared_mem", "occupancy limit (shmem)"),
    ("launch__occupancy_limit_blocks", "occupancy limit (blocks)"),
    ("sm__maximum_warps_per_active_cycle_pct", "theoretical occupancy [%]"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved occupancy [%]"),
    ("smsp__warps_eligible.avg.per_cycle_active", "eligible warps / cycle"),
    ("smsp__issue_active.avg.pct_of_peak_sustained_active", "issue active [%]"),
    ("dram__bytes_read.sum", "DRAM read bytes"),
    ("dram__bytes_write.sum", "DRAM write bytes"),
    ("dram__bytes_read.sum.pct_of_peak_sustained_elapsed", "DRAM read [% of peak]"),
    ("dram__bytes_write.sum.pct_of_peak_sustained_elapsed", "DRAM write [% of peak]"),
    ("lts__throughput.avg.pct_of_peak_sustained_elapsed", "L2 throughput [%]"),
    ("l1tex__throughput.avg.pct_of_peak_sustained_elapsed", "L1/tex throughput [%]"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM throughput [%]"),
    ("smsp__inst_executed.sum", "instructions"),
    ("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "global load requests"),
    ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "global load sectors"),
    ("l1tex__t_requests_pipe_lsu_mem_global_op_red.sum", "global atomic-add requests"),
    ("l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_atom.sum", "global atomic wavefronts"),
    ("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", "shared wavefronts"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", "shared bank conflicts"),
    ("l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum", "local (spill) load requests"),
    ("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", "stall: barrier"),
    ("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "stall: long scoreboard"),
    ("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", "stall: short scoreboard"),
    ("smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio", "stall: MIO throttle"),
    ("smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio", "stall: LG throttle"),
    ("smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", "stall: wait"),
    ("smsp__average_warps_issue_stalled_membar_per_issue_active.ratio", "stall: membar"),
    ("smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "stall: math pipe"),
    ("smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio", "stall: not selected"),
]


def variant_of(kernel_name):
    if "KokkosRestriction" in kernel_name or "KokkosProlongation" in kernel_name:
        return "bk1"
    if "Copy::" in kernel_name:
        return "copy"
    return "dealii"


def operation_of(kernel_name):
    return "R" if ("restrict" in kernel_name or "Restriction" in kernel_name) else "P"


def to_float(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def main(path):
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    units = rows[0] if rows and not to_float(rows[0][0] or "x") else None
    data = rows[1:] if units else rows
    col = {name: i for i, name in enumerate(header)}
    kname = col.get("Kernel Name")
    if kname is None:
        sys.exit("no 'Kernel Name' column: is this `ncu --csv --page raw` output?")

    groups = {}
    for r in data:
        name = r[kname]
        grid = r[col["launch__grid_size"]] if "launch__grid_size" in col else "?"
        key = (operation_of(name), variant_of(name), grid)
        groups.setdefault(key, []).append(r)

    # copy launches with the largest grid are the one-cell-per-team variant
    labels = {}
    for op in ("P", "R"):
        copies = sorted({k[2] for k in groups if k[0] == op and k[1] == "copy"}, key=lambda g: to_float(g) or 0)
        for i, grid in enumerate(copies):
            labels[(op, "copy", grid)] = "copy (1 cell/team)" if len(copies) > 1 and i == len(copies) - 1 else "copy"
    for k in groups:
        labels.setdefault(k, k[1])

    for op, opname in (("P", "prolongation"), ("R", "restriction")):
        keys = sorted((k for k in groups if k[0] == op), key=lambda k: ["dealii", "copy", "bk1"].index(k[1]) * 10 + (1 if labels[k].endswith("team)") else 0))
        if not keys:
            continue
        print(f"\n== {opname}: median over {', '.join(str(len(groups[k])) for k in keys)} launches ==")
        print(f"{'metric':<32}" + "".join(f"{labels[k]:>22}" for k in keys))
        for metric, label in METRICS:
            if metric not in col:
                continue
            vals = []
            for k in keys:
                xs = [to_float(r[col[metric]]) for r in groups[k]]
                xs = [x for x in xs if x is not None]
                vals.append(statistics.median(xs) if xs else None)
            if metric == "gpu__time_duration.sum":
                unit = units[col[metric]] if units else ""
                scale = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3}.get(unit, 1.0)
                vals = [v * scale if v is not None else None for v in vals]
            print(f"{label:<32}" + "".join(f"{v:>22,.1f}" if v is not None else f"{'-':>22}" for v in vals))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
