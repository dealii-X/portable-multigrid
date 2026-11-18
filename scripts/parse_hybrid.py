#!/usr/bin/env python3
"""Turn hybrid / bk3 output files into one CSV.

  parse_hybrid.py out1.out [out2.out ...] > summary.csv

A file is split into program runs at their "Settings of parameters:" blocks (one
per pair of thresholds when the hybrid program is given lists).
Hybrid runs (those with a "Ranks per GPU:" line) give one row per refinement
cycle; kind = cpu_only for the cycles that ran on the CPUs only.
poisson_cube_bk3 / poisson_cube_cpu runs (the "n_proc cells dofs mv_outer
mv_inner cg_time cg_its cg_reduction" table) give rows with kind = bk3 and
k = 0, switch_level = -1. Outputs of the hybrid program before the cycle loop
(one mesh per run, October 2026) are not handled any more.

CG times: min and median over the solves after the first one (the first is a
warm-up). Level times are the printed sums over the V-cycles of the last solve.
"""

import csv
import re
import statistics
import sys

NUM = r"[\d.]+(?:e[+-]?\d+)?"

SETTINGS = re.compile(r"^Settings of parameters:", re.M)
RANKS_PER_GPU = re.compile(r"^Ranks per GPU:\s+(\d+)", re.M)
DEGREE = re.compile(r"^Polynomial degree:\s+(\d+)", re.M)
N_SMOOTH = re.compile(r"^Number of pre-smoother iters:\s+(\d+)", re.M)
DOUBLING = re.compile(r"^Use doubling mesh:\s+(\d+)", re.M)
SWITCH_DOFS = re.compile(r"^Switch level, DoFs per rank:\s+(\d+)", re.M)
CPU_ONLY_DOFS = re.compile(r"^CPU-only cycle, DoFs per rank:\s+(\d+)", re.M)
CYCLE = re.compile(r"^Cycle (\d+)\s*$", re.M)
SWITCH_LEVEL = re.compile(r"^Switch level (\d+) of 0\.\.(\d+) with (\d+) DoFs \(", re.M)
CG = re.compile(
    r"CG on the fine level \((\d+) DoFs\): (\d+) iterations, reduction (" + NUM + r") per iteration, (" + NUM + r") s"
)
SWITCH = re.compile(
    r"==== switch, rank 0: (\d+) calls, bcast (" + NUM + r") device->host (" + NUM + r") CPU part (" + NUM +
    r") host->device (" + NUM + r")"
)
PREC = re.compile(r"Coarse solver (\d+) times: (" + NUM + r") tot prec (" + NUM + r")")
LEVEL = re.compile(r"^L(\d+)\s+(" + NUM + r")\s+(" + NUM + r")\s+(" + NUM + r")\s+(" + NUM + r")\s*$", re.M)
# n_proc is missing in poisson_cube_cpu's table
BK3_ROW = re.compile(
    r"^(?:(\d+)[ \t]+)?(\d+)[ \t]+(\d+)[ \t]+(" + NUM + r")[ \t]+(" + NUM + r")[ \t]+(" + NUM + r")[ \t]+(\d+)[ \t]+("
    + NUM + r")[ \t]*$", re.M
)

FIELDS = [
    "file", "kind", "k", "degree", "n_refine", "stretch", "cycle", "switch_level", "switch_level_dofs",
    "switch_dofs_per_rank", "cpu_only_dofs_per_rank", "n_smooth", "dofs", "its",
    "reduction", "cg_min", "cg_median", "gpu_prec", "cpu_prec", "cpu_part", "d2h", "h2d", "bcast",
    "gpu_levels", "cpu_levels",
]


def levels(block):
    """'L3:smoother/mv/restrict/prolongate;...' of one 'levels' table."""
    return ";".join(
        "L%s:%s/%s/%s/%s" % m.groups() for m in LEVEL.finditer(block)
    )


def hybrid_rows(name, run):
    """One row per refinement cycle of one hybrid program run."""
    k = int(RANKS_PER_GPU.search(run).group(1))
    degree = int(DEGREE.search(run).group(1))
    n_smooth = int(N_SMOOTH.search(run).group(1))
    doubling = DOUBLING.search(run).group(1) == "1"
    switch_dofs = int(SWITCH_DOFS.search(run).group(1))
    cpu_only = CPU_ONLY_DOFS.search(run)  # missing before the second threshold existed
    cpu_only_dofs = int(cpu_only.group(1)) if cpu_only else switch_dofs

    heads = list(CYCLE.finditer(run))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(run)
        part = run[h.start():end]
        cycle = int(h.group(1))
        cg = CG.findall(part)
        level = SWITCH_LEVEL.search(part)
        if not level:  # cycle below min_size or above max_size
            continue
        switch, fine = int(level.group(1)), int(level.group(2))
        if not cg:
            print("no CG result: %s k=%d cycle=%d" % (name, k, cycle), file=sys.stderr)
            continue
        times = [float(c[3]) for c in cg][1:] or [float(cg[0][3])]
        gpu_blk = cpu_blk = ""
        a, b, c = (part.find("==== GPU side levels"), part.find("==== CPU side levels"),
                   part.find("==== switch"))
        if a >= 0 and b > a:
            gpu_blk = part[a:b]
        if b >= 0:
            cpu_blk = part[b:c] if c > b else part[b:]
        sw = SWITCH.search(part)
        gp = PREC.search(gpu_blk)
        cp = PREC.search(cpu_blk)
        yield {
            "file": name, "kind": "cpu_only" if switch == fine else "hybrid", "k": k, "degree": degree,
            "n_refine": cycle // 3 if doubling else "", "stretch": cycle % 3 if doubling else -1,
            "cycle": cycle, "switch_level": switch, "switch_level_dofs": level.group(3),
            "switch_dofs_per_rank": switch_dofs, "cpu_only_dofs_per_rank": cpu_only_dofs,
            "n_smooth": n_smooth, "dofs": cg[0][0], "its": cg[0][1],
            "reduction": cg[0][2], "cg_min": min(times), "cg_median": statistics.median(times),
            "gpu_prec": gp.group(3) if gp else "", "cpu_prec": cp.group(3) if cp else "",
            "cpu_part": sw.group(4) if sw else "", "d2h": sw.group(3) if sw else "",
            "h2d": sw.group(5) if sw else "", "bcast": sw.group(2) if sw else "",
            "gpu_levels": levels(gpu_blk), "cpu_levels": levels(cpu_blk),
        }


def bk3_rows(name, text):
    seen = set()
    for m in BK3_ROW.finditer(text):
        n_proc, cells, dofs, _, _, cg_time, its, red = m.groups()
        if dofs in seen:  # the table is reprinted after every cycle
            continue
        seen.add(dofs)
        yield {
            "file": name, "kind": "bk3", "k": 0, "dofs": dofs, "its": its, "reduction": red,
            "cg_min": cg_time, "cg_median": cg_time, "switch_level": -1,
        }


def main(paths):
    out = csv.DictWriter(sys.stdout, fieldnames=FIELDS, restval="")
    out.writeheader()
    for path in paths:
        text = open(path, errors="replace").read()
        starts = [m.start() for m in SETTINGS.finditer(text)] or [0]
        for i, start in enumerate(starts):
            run = text[start:starts[i + 1] if i + 1 < len(starts) else len(text)]
            rows = hybrid_rows(path, run) if RANKS_PER_GPU.search(run) else bk3_rows(path, run)
            for row in rows:
                out.writerow(row)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
