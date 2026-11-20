# portable-multigrid

Matrix-free multigrid on GPUs built on deal.II's `Portable` (Kokkos) infrastructure:
Laplace operators with sum-factorization kernels (`include/operators`, `include/kernels`),
multigrid transfers (`include/multigrid`), solver drivers (`source`), CEED bake-off style
benchmarks (`tests`) and correctness tests (`correctness_tests`).

## Building

The programs need a deal.II installation built with Kokkos and CUDA. See
[docs/building_with_kokkos_cuda.md](docs/building_with_kokkos_cuda.md) for the Kokkos and
deal.II build recipe, the exit-time double free that a static Kokkos library causes, and
the JUPITER-specific launch conventions. Each driver and test is then a stand-alone CMake
project:

```bash
cmake -DDEAL_II_DIR=<deal.II install prefix> -DCMAKE_BUILD_TYPE=Release -S tests/poisson_cube_bk3 -B tests/poisson_cube_bk3/build
make -C tests/poisson_cube_bk3/build -j
```
