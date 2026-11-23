# portable-multigrid

Matrix-free multigrid on GPUs built on deal.II's `Portable` (Kokkos) infrastructure:
Laplace operators with sum-factorization kernels (`include/operators`, `include/kernels`),
multigrid transfers (`include/multigrid`), solver drivers (`source`), benchmarks
(`performance_tests`) and the testsuite (`tests`).

## Building

The programs need a deal.II installation built with Kokkos and CUDA. See
[docs/building_with_kokkos_cuda.md](docs/building_with_kokkos_cuda.md) for the Kokkos and
deal.II build recipe, the exit-time double free that a static Kokkos library causes, and
the JUPITER-specific launch conventions. Then, from the repository root:

```bash
cmake -S . -B build -DDEAL_II_DIR=<deal.II install prefix> -DCMAKE_BUILD_TYPE=Release
cmake --build build -j<N>
ctest --test-dir build -j<N>
```

Every directory in `source/` and `performance_tests/` also builds on its own:

```bash
cmake -S performance_tests/poisson_cube_bk3 -B performance_tests/poisson_cube_bk3/build -DDEAL_II_DIR=<deal.II install prefix> -DCMAKE_BUILD_TYPE=Release
cmake --build performance_tests/poisson_cube_bk3/build -j<N>
```
