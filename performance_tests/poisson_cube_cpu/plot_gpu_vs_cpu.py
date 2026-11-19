#!/usr/bin/env python3
"""Parse the result files of run_gpu_vs_cpu.sh and plot throughput (DoF/s).

usage: plot_gpu_vs_cpu.py <jobid> [results_dir] [override_jobid]
reads  <results_dir>/<label>.<jobid>.out for the labels below; when an
       override job id is given, a label whose file exists for that job id
       is taken from there instead (e.g. socket-bound CPU reruns)
writes <results_dir>/gpu_vs_cpu.<jobid>[+<override_jobid>].png and .csv
"""
import re
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

jobid = sys.argv[1]
rdir = sys.argv[2] if len(sys.argv) > 2 else "."
override = sys.argv[3] if len(sys.argv) > 3 else None
tag = f"{jobid}+{override}" if override else jobid

RUNS = [  # label, legend, color, linestyle
    ("gpu64.np1",   "GPU bk3: 1 H100 (fp64)",                "tab:red",    "-"),
    ("gpu64.np2",   "GPU bk3: 2 H100 (fp64)",                "tab:red",    "--"),
    ("gpu64.np4",   "GPU bk3: 4 H100 = node (fp64)",         "tab:red",    ":"),
    ("cpu64.np72",  "CPU: 72 cores = 1 Grace (fp64)",        "tab:blue",   "-"),
    ("cpu64.np144", "CPU: 144 cores = 2 Grace (fp64)",       "tab:blue",   "--"),
    ("cpu64.np288", "CPU: 288 cores = node (fp64)",          "tab:blue",   ":"),
    ("gpu32.np1",   "GPU bk3: 1 H100 (fp32 V-cycle)",        "tab:orange", "-"),
    ("gpu32.np2",   "GPU bk3: 2 H100 (fp32 V-cycle)",        "tab:orange", "--"),
    ("gpu32.np4",   "GPU bk3: 4 H100 = node (fp32 V-cycle)", "tab:orange", ":"),
    ("cpu32.np72",  "CPU: 72 cores (fp32 V-cycle)",          "tab:green",  "-"),
    ("cpu32.np144", "CPU: 144 cores (fp32 V-cycle)",         "tab:green",  "--"),
    ("cpu32.np288", "CPU: 288 cores (fp32 V-cycle)",         "tab:green",  ":"),
]
LINE = re.compile(r"^Best timings for ndof =\s*(\d+)\s+mv\s+([0-9.e+-]+)\s+mv smooth\s+([0-9.e+-]+)\s+cg-mg\s+([0-9.e+-]+)")

data = {}
for label, *_ in RUNS:
    path = os.path.join(rdir, f"{label}.{jobid}.out")
    if override and os.path.exists(os.path.join(rdir, f"{label}.{override}.out")):
        path = os.path.join(rdir, f"{label}.{override}.out")
        print(f"{label}: taken from job {override}")
    if not os.path.exists(path):
        print(f"missing: {path}")
        continue
    rows = []
    with open(path) as f:
        for line in f:
            m = LINE.match(line)
            if m:
                n, mv, mvs, cg = int(m[1]), float(m[2]), float(m[3]), float(m[4])
                rows.append((n, mv, mvs, cg))
    data[label] = rows
    print(f"{label}: {len(rows)} sizes, largest {rows[-1][0] if rows else 0}")

with open(os.path.join(rdir, f"gpu_vs_cpu.{tag}.csv"), "w") as f:
    f.write("run,ndofs,mv_s,mv_smooth_s,cg_mg_s,mv_dofs_per_s,smooth_dofs_per_s,cg_mg_dofs_per_s\n")
    for label, rows in data.items():
        for n, mv, mvs, cg in rows:
            f.write(f"{label},{n},{mv},{mvs},{cg},{n/mv:.4g},{n/mvs:.4g},{n/cg:.4g}\n")

def units_of(label):  # number of GH200 superchips used: np<k> GPUs, or np<72k> cores
    n = int(label.split(".np")[1])
    return n // 72 if n >= 72 else n

for norm, suffix, subtitle, yscale in [
        (False, "", "absolute", "log"),
        (True, ".per_superchip", "per GH200 superchip", "log"),
        (False, ".semilog", "absolute", "linear"),
        (True, ".per_superchip.semilog", "per GH200 superchip", "linear")]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharex=True)
    panels = [  # column index in rows, title, y scale factor, y unit
        (1, "matrix-vector product (fp64)", 1e-9, "GDoF"),
        (2, "smoother matrix-vector product", 1e-9, "GDoF"),
        (3, "CG with multigrid: full solve", 1e-6, "MDoF"),
    ]
    for ax, (idx, title, scale, unit) in zip(axes, panels):
        for label, legend, color, ls in RUNS:
            rows = data.get(label)
            if not rows:
                continue
            u = units_of(label) if norm else 1
            xs = [r[0] / u for r in rows]
            ys = [r[0] / r[idx] / u * scale for r in rows]
            ax.plot(xs, ys, marker="o", ms=3, color=color, ls=ls, lw=1.8, label=legend)
        ax.set_title(title)
        ax.set_xlabel("DoFs / chip" if norm else "DoFs")
        ax.set_ylabel(f"{unit} / [s x chip]" if norm else f"{unit} / s")
        ax.set_xscale("log")
        ax.set_xlim(left=5e3)  # drop the latency-dominated small sizes
        ax.set_yscale(yscale)
        if yscale == "log":
            # plain decimal tick labels on the y axis: 0.01, 0.1, 1, 10, 100
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_ylim(bottom=0)
        ax.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Poisson, Q4, 3D, 5+5 Chebyshev smoothing, GH200 node of JUPITER, deal.II dealii-neon (job {tag}), {subtitle}; chip = 1 H100 or 72 Grace cores")
    fig.tight_layout(rect=(0, 0.07, 1, 1))  # leave room for the legend below the panels
    out = os.path.join(rdir, f"gpu_vs_cpu.{tag}{suffix}.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)

# latency figures: throughput against wall time per operation, each curve parametrized by size;
# once log-log and once semilog (linear throughput from zero)
for yscale, suffix in [("log", ".latency"), ("linear", ".latency.semilog")]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    panels = [(1, "matrix-vector product (fp64)", 1e-9, "GDoF"),
              (2, "smoother matrix-vector product", 1e-9, "GDoF"),
              (3, "CG with multigrid: full solve", 1e-6, "MDoF")]
    for ax, (idx, title, scale, unit) in zip(axes, panels):
        for label, legend, color, ls in RUNS:
            rows = data.get(label)
            if not rows:
                continue
            xs = [r[idx] * 1e3 for r in rows]
            ys = [r[0] / r[idx] * scale for r in rows]
            ax.plot(xs, ys, marker="o", ms=3, color=color, ls=ls, lw=1.8, label=legend)
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.set_title(title)
        ax.set_xlabel("time per operation [ms]")
        ax.set_ylabel(f"{unit} / s")
        axes_to_format = (ax.xaxis, ax.yaxis) if yscale == "log" else (ax.xaxis,)
        for axis in axes_to_format:
            axis.set_major_locator(LogLocator(base=10, numticks=12))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
        if yscale == "linear":
            ax.set_ylim(bottom=0)
        ax.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Poisson, Q4, 3D, 5+5 Chebyshev smoothing, GH200 node of JUPITER, deal.II dealii-neon (job {tag}): throughput against time per operation, points = problem sizes")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = os.path.join(rdir, f"gpu_vs_cpu.{tag}{suffix}.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)

# node-level figure: the four full-node runs only (4 H100 vs 288 Grace cores), total DoFs
NODE_RUNS = [
    ("gpu64.np4",   "GPU node: 4 H100, fp64 V-cycle",          "tab:red",    "-"),
    ("gpu32.np4",   "GPU node: 4 H100, fp32 V-cycle",          "tab:orange", "-"),
    ("cpu64.np288", "CPU node: 288 Grace cores, fp64 V-cycle", "tab:blue",   "-"),
    ("cpu32.np288", "CPU node: 288 Grace cores, fp32 V-cycle", "tab:green",  "-"),
]
for yscale, suffix in [("log", ".node"), ("linear", ".node.semilog")]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    panels = [(1, "matrix-vector product (fp64)", 1e-9, "GDoF"),
              (2, "smoother matrix-vector product", 1e-9, "GDoF"),
              (3, "CG with multigrid: full solve", 1e-6, "MDoF")]
    for ax, (idx, title, scale, unit) in zip(axes, panels):
        for label, legend, color, ls in NODE_RUNS:
            rows = data.get(label)
            if not rows:
                continue
            xs = [r[0] for r in rows]
            ys = [r[0] / r[idx] * scale for r in rows]
            ax.plot(xs, ys, marker="o", ms=4, color=color, ls=ls, lw=2.2, label=legend)
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        if yscale == "linear":
            ax.set_xlim(left=5e3)  # the log-log node figure keeps the full size range
        ax.set_title(title)
        ax.set_xlabel("DoFs")
        ax.set_ylabel(f"{unit} / s")
        if yscale == "log":
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_ylim(bottom=0)
        ax.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Poisson, Q4, 3D, 5+5 Chebyshev smoothing, one GH200 node of JUPITER (4 H100 + 4 x 72 Grace cores), deal.II dealii-neon (job {tag})")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = os.path.join(rdir, f"gpu_vs_cpu.{tag}{suffix}.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)
