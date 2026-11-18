# Hybrid CPU-GPU multigrid: plan for the next test round

Written 2026-10-14, before a three-week break. Read "State of play" and "First day back" first.

## State of play

Code: `tests/poisson_cube_hybrid_gpu_cpu/`, branch `cpu-tests`, last commit `b1f11bc`.
Design D2: k ranks per GPU (1 GPU rank + k-1 CPU-only). GPU ranks run levels `s..M`, all ranks run the CPU
V-cycle on levels `0..s` (`s` = `switch_level`, level `s` exists on both sides). Permutation P moves level `s`.

CLI (since 2026-11-12): `./program k degree min_size max_size n_pre n_post mesh switch_dofs`, the arguments of
`poisson_cube_bk3` with `k` in front and `switch_dofs` at the end. The program loops over the refinement cycles
like bk3 and prints bk3's tables after each. Per cycle the switch level is the finest level with at most
`switch_dofs` DoFs per MPI rank of comm_world (0 = all levels on the GPUs); if that is the fine level, the
cycle runs on the CPUs only.
The results below were obtained with the old CLI `k degree n_refine switch_level n_subdivisions [n_smooth]
[stretch]` (one mesh per run, explicit switch level).

Verified on JUPITER (1 node, 4 GH200, `dealii-mg-transfer`, CPU side scalar = no NEON):

| Check | Result |
|---|---|
| CG iterations / reduction | identical to bk3 and CPU-only for every k, cycle and switch level |
| P (GPU<->CPU permutation) | exact, 4e-16 |
| k=1/4, `switch_level=0` vs bk3 | same time within 2-4% |
| 24^3 mesh, Q4, cheb 3: best | k=1 s=2 0.0471 s, k=4 s=4 **0.0394 s** vs bk3 0.0600 s (1.49x) |
| doubling cycles 12-20, Q4, cheb 5/5 | k=4 s=4 is 1.63x (275k DoFs) ... 1.21x (68M DoFs) faster than bk3; saving ~constant 22 ms |
| k=18 | much worse; fine GPU level 3x slower, gap grows with size (open) |

Results: `results/hybrid-2320320.out` (24^3 sweep), `results/doubling_mesh/` (job 2320751 + `gpu_baseline.out`).
Launch recipe that works: `srun --ntasks=4k --cpus-per-task=72/k --distribution=block:block --cpu-bind=cores
--gpu-bind=closest --mem-bind=local` (never with `--ntasks-per-socket`). Details in `scripts/*.sbatch`.

Open questions this plan answers:

1. Where exactly is the GPU/CPU crossover as a function of DoFs per module, degree, k and CPU speed (NEON)?
2. Why does k=18 slow down the fine GPU level?
3. How does the hybrid compare with GPU-only across sizes, degrees, smoothers and node counts?
4. Is it better to offload coarse levels or to make the GPU coarse levels cheaper (the baseline to beat)?

A clue for question 4, from bk3's own timings (343-DoF level, per rank): ghost exchange 0.126 ms, compute only
0.043 ms. On the CPU the same ghost exchange is 0.010 ms. The GPU latency floor looks dominated by
communication, not kernel launch. Phase 2 tests this.

## First day back

1. Read this file, `results/`, and the memory notes (hybrid + NEON).
2. Pull, rebuild `bk3`, `poisson_cube_cpu`, hybrid. Rerun `scripts/hybrid_baselines_jupiter.sbatch` (5 min) to
   confirm nothing changed on the machine (modules, deal.II).
3. Decide the NEON status (Phase 1.2) before any new CPU-side timing.

## Phase 1: housekeeping (1-2 days)

| # | Task | Done when |
|---|---|---|
| 1.1 | Record the exact software stack (modules, deal.II install + commit, Kokkos version) in the results folder | one `STACK.txt` |
| 1.2 | NEON deal.II (`dealii-neon`, branch `neon-vector-extensions`): `fmla ... .2d` count in `libdeal_II.so`; `nm program \| grep VectorizedArray` shows 2-wide double; float/double matvec ratio ~0.5 not ~1; build hybrid and `poisson_cube_cpu` against it; rerun the 3 baselines | CG its identical, CPU matvec faster than scalar |
| 1.3 | Kokkos as shared library (removes exit abort, double free of static map) or keep the stderr filter | clean exit, or accepted |
| 1.4 | `scripts/parse_hybrid.py` (written, tested on the existing results: `parse_hybrid.py out... > summary.csv`, handles hybrid and bk3 outputs); still to do: a plot script | reproduces the tables in this file |
| 1.5 | Job template: separate `.err`, per-run `timeout`, placement print, `STACK.txt` header | used by all later jobs |

## Phase 2: microbenchmarks for the crossover (the core study)

All results go to CSV with: machine, ranks, ranks per Grace, DoFs per rank, degree, precision, time, repeats.
Use medians of >= 10 repeats; first call discarded.

### 2A. Matvec throughput, GPU vs CPU

Operator: matrix-free Laplace, Q1/Q2/Q3/Q4, double (and float for the V-cycle).
The existing programs already print compute-only, ghost-only and ghost+compute matvec times per level
(`Best timings for ndof = ...`), so first reuse `poisson_cube_bk3` and `poisson_cube_cpu`; add a standalone driver
only if a size or rank count is missing.

| Axis | Values |
|---|---|
| DoFs per module | 1e2 ... 1e8, log-spaced (doubling meshes) |
| GPU | 1 rank/GPU, 4 GPUs together and 1 GPU alone (isolates communication) |
| CPU ranks per Grace | 1, 2, 4, 8, 18, 36, 72 (72 % r == 0), scalar vs NEON |
| Load | 1 Grace active vs all 4 active (memory bandwidth, power contention) |
| Also | Chebyshev smoother step (degree 3, 5), restriction, prolongation per level |

Outputs: time per matvec and DoFs/s vs DoFs per module for GPU and CPU (per Grace, all cores). Fit
`t(n) = t0 + n / R` for each; the intersection `n*(degree, r, NEON)` is the crossover. Prediction: the best
`switch_level` is the finest level with fewer than `n*` DoFs per module.

### 2B. Communication

Ready to run: `tests/comm_microbench/` (MPI + Kokkos, tested locally) and the job `scripts/comm_microbench_jupiter.sbatch`
(k = 1, 4, 18; host / staged / device buffers; pingpong, halo, allreduce; spin vs sleep idle ranks). Build it
like the other tests, submit from its build dir. Coverage:

| Test | Variants |
|---|---|
| Ping-pong latency and bandwidth, 8 B ... 64 MB | host vs device buffers; same Grace, different Grace same node, different nodes |
| Halo exchange (6 face neighbours, realistic sizes) | host vs device; 4 GPU ranks vs 16 vs 72 total ranks in the job |
| Allreduce latency (CG dot products) | 4, 16, 72 ranks; 1 and 2 nodes |
| Idle-rank effect (k=18 anomaly) | same test while the extra ranks are idle in `MPI_Bcast`, with `OMPI_MCA_mpi_yield_when_idle=1`, and while not initialising CUDA |
| Switch cost parts | D2H, P (exchange), bcast, H2D vs level size (already printed by hybrid) |

Also record: `ompi_info` (CUDA-aware?), UCX transports (`ucx_info`), env used. The k=18 diagnostics:
k = 2, 4, 8, 12, 18 at fixed cycle 18, `switch_level=0`; yield-when-idle; MPS if available.

### 2C. GPU latency floor

| Test | Purpose |
|---|---|
| Empty `parallel_for` + `fence` latency; chain of N tiny kernels | pure launch cost |
| One Chebyshev step on a 343 / 2k / 15k-DoF level, no MPI (single GPU) | kernels only |
| Same with the 4-GPU halo exchange | communication share |
| `nsys` timeline of one V-cycle at a coarse level | where the ~0.9 ms per level per V-cycle goes |

Decision rule: if communication dominates, test cheaper GPU halo exchange (fewer or fused packs, CUDA-aware path,
agglomerating coarse levels onto fewer GPU ranks) as an alternative to offloading. Keep that variant as a
second baseline, "GPU-only with optimised coarse levels", so the hybrid is compared fairly.

### 2D. Cost model vs measurements

Predict the best `s` and the speedup from 2A-2C for every (size, k) already measured (24^3 sweep, doubling
cycles 12-20) and compare. Deliverable: one plot of predicted vs measured CG time and best `s`.

## Phase 3: full hybrid test matrix (after Phase 2 fixes the ranges)

| Axis | Values |
|---|---|
| Mesh | doubling cycles up to the GPU memory limit (bk3 reached cycle 22 = 270M DoFs, out of memory at 540M) |
| Degree | Q2, Q3, Q4 (the CLI supports 1-4) |
| Chebyshev | 3/3 and 5/5 |
| k | 1, 2, 4, 6, 8, 9, 12 (+18 once understood) |
| `switch_level` | 0 ... L-1, including the p-levels (Q2 / Q1 levels of the fine mesh) |
| CPU build | scalar and NEON |
| Nodes | 1 first; then 2, 4, 8, 16 nodes: strong scaling (fixed mesh) and weak scaling |

Multi-node hybrid has only been tested with simulated nodes (pc02), never on JUPITER. Do 2 nodes with a small
mesh first, check the layout table (one GPU rank per Grace, blocks inside a node) and the regression.

Per run record: CG its and reduction (must match bk3), CG time min/median, per-level GPU and CPU times,
switch breakdown, setup time, host and HBM memory per rank (HBM per CUDA context of CPU-only ranks is
still unmeasured). Always run the `switch_level=0` and bk3 reference in the same job to remove
node-to-node variation.

Budget: a hybrid run takes 0.5-2 min at the sizes above. Group runs in jobs of <= 1 h, k=4 first, and put the
cheapest informative runs first so a timeout loses little.

## Phase 4: code work, only if the data asks for it

| Item | Trigger |
|---|---|
| Fix the k=18 effect (idle ranks, CUDA contexts of CPU-only ranks, MPS) | Phase 2B identifies the cause |
| Choose `switch_level` automatically from the cost model (DoFs per module < n*) | Phase 2D model is within ~10% |
| GPU-only coarse-level agglomeration baseline | Phase 2C says communication dominates |
| CPU V-cycle in single precision / further CPU tuning | CPU V-cycle remains the largest share at k=4 (it is 22 of 39 ms at 912k DoFs, `switch_level=4`) |
| Multi-node hybrid fixes | first 2-node run |

## Reporting

Plots: speedup vs DoFs per GPU (per degree), crossover `n*` vs (k, NEON), predicted vs measured, scaling.
Feed the NHR proposal GPU part (`project_nhr_extension_proposal`, Program B).

## Parallel track, not started

BDDC static condensation follow-ups (tasks #11-#14 in the project memory): fuse the Lagrange recombination,
block adapters, batched Schur setup. #14 (use static condensation inside the coarse-matrix setup) needs your
review of the math before any code. Details: `project_static_condensation_arr.md` in the memory folder.

## Risks

- The JUPITER software stack may change during the break (modules, Slurm): rerun the baselines first.
- `dealii-neon` may not match the branch the hybrid needs; fallback is scalar CPU numbers.
- Queue wait; use short jobs first.
- Do not run `make` in a directory while a job uses its binary; use a second build directory.
- Every SSH or scp needs an explicit go-ahead from you.

## Status after the November 2026 rounds, and what is next

Written 2026-11-12. Details and numbers: results folders `results_jupiter*/` and the outputs listed there.

What the level-split hybrid is: a latency fix. It moves the levels that are too small for the GPUs to the
CPUs. Its saving per solve is fixed (20 - 27 ms on one node, 28 - 30 ms on two), so the gain over GPU-only
fades as the fine level grows: 2x at 275k DoFs, 1.2x at 68M, about 1.07x expected at 270M (one node).
Thresholds, both per MPI rank and roughly independent of k and of the node count: switch level at 2000 - 8000
DoFs, CPU-only cycle up to about 50000 fine-level DoFs.

### Near term

| # | Item | Why |
|---|---|---|
| N1 | Idle CPU-only ranks wait without spinning. Low priority after job 2880860 | Cause confirmed: busy neighbouring cores slow the GPU rank's host side (GPU-only program with busy loops on the other cores: +6 - 9 ms per solve with 70 busy cores per socket, +8 - 12 ms with 71; the number of cores of the GPU rank itself does not matter). This covers the slowdown at k=36 and about half of it at k=72. But the extra time is spread evenly over the levels, and the hybrid keeps only 1 - 3 of them on the GPUs: there it is 1 - 2 ms at k=36 and 1 - 5 ms at k=72, so the hybrid would gain 1 - 4%. |
| N2 | Use fewer ranks than k for the CPU levels of a hybrid cycle | The CPU part is latency-bound as well: the same 6 levels cost 11.1 / 9.2 / 8.8 ms at k = 12 / 18 / 24. Best k for hybrid cycles is 18 - 24, for CPU-only cycles 72; with N1 one launch could serve both. |
| N3 | Done 2026-11-13: overnight runs of 2026-11-12 evaluated | Gain over GPU-only at equal DoFs per GPU is the same on 1, 2, 4 nodes at large sizes (34M: 1.11 - 1.13x, 67M: 1.04 - 1.07x). CPU-only is fastest up to about 60000 DoFs per core on every node count, the hybrid above. k = 18 and 36 are equivalent, k = 72 is never the best hybrid. |

### Long term

| # | Item | Why |
|---|---|---|
| L1 | Weighted split of the fine levels between CPU and GPU ranks | In the throughput-bound regime the CPU cores idle while the GPUs smooth the fine level. One Grace socket has about a third of the matvec throughput of one H100, so sharing the fine level by weight has a ceiling of about 1.3 - 1.4x over GPU-only. Needs weighted partitioning, ghost exchange between host and device vectors on one level, and a smoother across both. Moving more levels to the CPUs is not an alternative: one level above the switch already costs 20 - 30%. |
| L2 | Move the hybrid test onto the refactor-enes base | Its bk3 has 6.5 ms less latency per solve (constant); the hybrid would gain about 2 ms, GPU-only all of it. |
| L3 | CPU-only ranks from a host-only build (`srun --multi-prog`) | Start-up costs 1.2 s per process and node (CUDA runtime), 6 min at k=72; MPS does not help and allows at most 60 processes per GPU. |
