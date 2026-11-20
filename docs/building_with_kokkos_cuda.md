# Building deal.II with Kokkos + CUDA for this repository

This note documents how to build the Kokkos / deal.II stack that the programs in this
repository need on an NVIDIA GPU machine, and the one configuration mistake that is easy
to make and hard to diagnose: a **double free at process exit** in every program that
initializes Kokkos. The instructions were verified on JUPITER (GH200, aarch64) on
2026-11-10 with the software stack listed at the end, but nothing in them is specific
to that machine.

Short version:

1. Build Kokkos as a **shared** library (`-DBUILD_SHARED_LIBS=ON`).
2. Build deal.II against that Kokkos, with `mpicxx` wrapping the `nvcc_wrapper` **of the
   same Kokkos installation**, in a fresh build directory.
3. Build the programs here with `-DDEAL_II_DIR=<that install>`; no special link flags
   are needed.

## 1. The exit-time double free

### Symptom

Every CUDA program linked against deal.II aborts *after* `main()` has returned:

```text
free(): double free detected in tcache 2        # or: double free or corruption (!prev)
```

The computation itself is correct; only the process exit code (134, `SIGABRT`) is wrong.
deal.II's own `quick_tests/kokkos` fails the same way, so this is not a bug in this
repository. Programs that never initialize Kokkos (no `MPI_InitFinalize`, no
`Kokkos::initialize`) are unaffected, which makes the problem look intermittent.

### Cause

Two copies of the Kokkos runtime end up in one process:

- Kokkos is built as a static library by default (`libkokkoscore.a`), and deal.II exports
  it as such: `dealii::interface_kokkos` carries the full paths of the `.a` files.
- deal.II itself is linked with `nvcc_wrapper`. `nvcc_wrapper` cannot pass versioned shared
  objects (`*.so.*`) to `nvcc` as inputs, so it takes them out of the argument list and
  hands them to the host linker as `-Xlinker` options (Kokkos 4.7/5.x
  `bin/nvcc_wrapper`, the `*.so.*` case). On the final host link line
  `libdeal_II.so.<version>` therefore comes **after** `libkokkoscore.a`, the reverse of
  the order CMake asked for.
- With the archive first, the linker pulls Kokkos's object files into the executable to
  satisfy the program's own references. `libdeal_II.so` already contains an identical
  copy. Both images run their static constructors and destructors on the same
  (interposed) objects, and the second `__cxa_finalize` frees memory the first one
  already freed.

You can confirm it on any affected executable; the counts below are from
`quick_tests/kokkos.release` of a deal.II master build with static Kokkos 4.7:

```bash
# Kokkos symbols compiled into the executable
nm -C --defined-only kokkos.release | grep -c "Kokkos::"                       # 619
# ... of which also exported by libdeal_II.so  (= the duplicates)
comm -12 <(nm -C --defined-only --extern-only kokkos.release | cut -d' ' -f3- | sort -u) \
         <(nm -C -D --defined-only libdeal_II.so | cut -d' ' -f3- | sort -u) | grep -c "Kokkos::"   # 409
```

Anything above zero in the second count is the bug. (Weak template instantiations
are legitimately duplicated; `--extern-only` strong definitions in both images are
not.)

### The link-order workaround, and why it is only that

Enes Soydan found the mechanism while auditing the CEED BP3 benchmark
([dealii-X/benchmarks#13](https://github.com/dealii-X/benchmarks/pull/13), "Bug 6") and
worked around it in the benchmark's `CMakeLists.txt`: link the *unversioned*
`libdeal_II.so` explicitly **before** calling `deal_ii_setup_target()`. The plain `.so`
name matches `nvcc_wrapper`'s ordinary input pattern, stays ahead of `libkokkoscore.a`,
and the executable then resolves every Kokkos symbol from the shared library instead of
extracting archive members. It works, and it was the key to understanding the problem,
but it is a workaround rather than a fix:

- it has to be repeated in the CMake file of every executable (this repository alone has
  dozens of drivers and correctness tests);
- it depends on an internal detail of `nvcc_wrapper`'s argument sorting;
- it does nothing for deal.II's own test suite, which links its targets in-tree;
- the executable still carries a second, partial copy of Kokkos for any symbol the
  shared library happens not to export.

### The fix

Build Kokkos as a shared library. There is then exactly one `libkokkoscore.so` in the
process, loaded once by the dynamic loader and used by `libdeal_II.so` and the executable
alike, and the link order no longer matters. With this, deal.II's quick tests pass,
`MPI_InitFinalize`-based programs exit with 0, and the duplicate count above is 0.
(Building deal.II itself statically, `-DBUILD_SHARED_LIBS=OFF`, also removes the
duplication, at the cost of very large executables.)

## 2. Kokkos

Kokkos 5.x requires C++20, CMake >= 3.22, NVCC >= 12.2 and GCC >= 10.4. deal.II `master`
supports Kokkos 5.0 since [dealii/dealii#19045](https://github.com/dealii/dealii/pull/19045)
(2025-12) and 5.2+ since [dealii/dealii#19456](https://github.com/dealii/dealii/pull/19456)
(2026-03-27); use a deal.II checkout newer than that, or Kokkos 4.7.04 with the same
recipe.

```bash
git clone --depth 1 --branch 5.2.2 https://github.com/kokkos/kokkos.git kokkos-5.2.2
mkdir kokkos-5.2.2/build && cd kokkos-5.2.2/build
export CC=gcc CXX=g++        # Kokkos switches to its own nvcc_wrapper when CUDA is enabled
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DBUILD_SHARED_LIBS=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
  -DKokkos_ARCH_HOPPER90=ON \
  -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF \
  -DCMAKE_INSTALL_PREFIX=$KOKKOS_PREFIX
make -j install
```

- `BUILD_SHARED_LIBS=ON` is the fix from section 1. `CMAKE_POSITION_INDEPENDENT_CODE` is
  then unnecessary.
- `Kokkos_ARCH_HOPPER90` is for GH200/H100; pick the architecture of your GPU.
- `Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF`: stream-ordered allocations cannot be
  registered for RDMA by UCX, which breaks device-aware MPI. Explicitly off is safe
  whatever the default of your Kokkos version.

## 3. deal.II

deal.II must be compiled by the `nvcc_wrapper` **of the Kokkos installation it is built
against** (the wrapper carries the default `-arch` and the argument handling of its Kokkos
version). With OpenMPI the compiler wrapper is redirected through `OMPI_CXX`; with
ParaStationMPI / MPICH-based stacks the variable is `MPICH_CXX`.

```bash
export OMPI_CXX=$KOKKOS_PREFIX/bin/nvcc_wrapper OMPI_CC=gcc      # OpenMPI
# export MPICH_CXX=$KOKKOS_PREFIX/bin/nvcc_wrapper MPICH_CC=gcc  # ParaStationMPI / MPICH
export LD_LIBRARY_PATH=$KOKKOS_PREFIX/lib64:$LD_LIBRARY_PATH

mkdir dealii/build-cuda && cd dealii/build-cuda     # a FRESH build directory, see below
cmake ../dealii \
  -DCMAKE_INSTALL_PREFIX=$DEALII_PREFIX \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_Fortran_COMPILER=mpifort \
  -DMPI_C_COMPILER=mpicc -DMPI_CXX_COMPILER=mpicxx -DMPI_Fortran_COMPILER=mpifort \
  -DKokkos_DIR=$KOKKOS_PREFIX/lib64/cmake/Kokkos \
  -DDEAL_II_WITH_MPI=ON \
  -DDEAL_II_MPI_WITH_DEVICE_SUPPORT=ON \
  -DDEAL_II_WITH_64BIT_INDICES=ON \
  -DDEAL_II_WITH_P4EST=ON -DP4EST_DIR=$P4EST_PREFIX \
  -DDEAL_II_WITH_LAPACK=ON
make -j install
```

- **Fresh build directory.** deal.II caches the Kokkos library list; reconfiguring an
  existing build directory with a new `Kokkos_DIR` keeps the old `.a` paths.
- Check the configuration before the long compile: `detailed.log` in the build directory
  must show `KOKKOS_VERSION = <your version>` and `KOKKOS_BACKENDS = ...;CUDA`, and
  `$KOKKOS_PREFIX/lib64` must contain only `libkokkoscore.so*`, no `libkokkoscore.a`.
- The harmless message `nvcc_wrapper has been given GNU extension standard flag
  -std=gnu++20 - reverting flag to -std=c++20` appears for every translation unit.
- `make test` runs deal.II's quick tests. The serial ones (including `kokkos`) need a
  GPU on the node where `make test` runs; the MPI ones are launched through the
  `mpiexec` CMake detected. On a Slurm system where that launcher cannot start nested
  jobs (see section 5), run the two binaries directly instead:
  `srun -n 2 tests/quick_tests/mpi.release/mpi.release` and
  `srun -n 10 tests/quick_tests/p4est.release/p4est.release`; both must exit with 0.

## 4. Building and running the programs of this repository

Every driver and correctness test is a stand-alone CMake project:

```bash
cmake -DDEAL_II_DIR=$DEALII_PREFIX -DCMAKE_BUILD_TYPE=Release -S tests/poisson_cube_bk3 -B tests/poisson_cube_bk3/build
make -C tests/poisson_cube_bk3/build -j
```

No link-order tricks are required once deal.II is built as described above. If a program
still aborts at exit, check it with the `nm`/`comm` commands of section 1 and
`ldd ./program | grep kokkos` (which must list `libkokkoscore.so`).

The Kokkos OpenMP backend prints an `OMP_PROC_BIND` warning per rank at startup; it is
cosmetic. `export OMP_PROC_BIND=spread OMP_PLACES=threads` (or `srun --cpu-bind=cores`)
silences it.

## 5. JUPITER specifics

Verified stack (Stage 2026): `nvidia-compilers/25.9-CUDA-13`, `OpenMPI/5.0.8` with
`MPI-settings/CUDA` and `UCX-settings/RC-CUDA`, `FlexiBLAS/3.4.5`, Kokkos 5.2.2,
deal.II master (2026-11), p4est 2.8.7, CMake from the stage.

- LAPACK comes from FlexiBLAS: add
  `-DLAPACK_FOUND=true -DLAPACK_DIR=$EBROOTFLEXIBLAS -DBLAS_DIR=$EBROOTFLEXIBLAS` to the
  deal.II configure line; `nvfortran` works as the Fortran compiler.
- Build on a compute node (`salloc` + `srun -N1 -n1 -c96 make -j96 install`; a GH200 node
  has 288 cores but each `nvcc` translation unit needs 1-3 GB, so do not use them all).
- Launch MPI programs with `srun -n <ranks> ./program` **from the `salloc` shell on the
  login node**, one job step per run. There is no `mpiexec` on the path, and a launcher
  started inside an interactive `srun --pty` step cannot create a nested partition
  (`PSI: PSI_createPartition: got: Permission denied`, or a hang with `--overlap`).
- Compute nodes have no internet access: `git clone`/`git pull` on the login node.

`run_sbatch.sh` in the repository root shows the batch-job form of the same launch
(`srun --cpu-bind=cores --gpu-bind=closest --gpus-per-task=1 ./program ...`).
