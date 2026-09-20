cmake_minimum_required(VERSION 3.13.4)

# 1. Set the target name
set(TARGET "program")

# 2. Source files
set(TARGET_SRC ${TARGET}.cc)

# 3. Find deal.II
#
# vmult_tensor_core() (bk4_cuda_kernels.cuh) only does anything useful when
# this ends up pointed at a deal.II built against a CUDA-enabled Kokkos
# (KOKKOS_ENABLE_CUDA=ON) -- with a Serial/OpenMP-only Kokkos, program.cc
# still builds and runs, it just skips the tensor-core comparison and only
# checks vmult_bk4() vs. vmult_dealii(). deal.II's own CMake macros
# transitively pull in whatever nvcc flags Kokkos+CUDA needs (--extended-
# lambda, --expt-relaxed-constexpr) for compiling program.cc, so nothing
# extra needs to be added here for that.
find_package(deal.II 9.8.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "Could not locate deal.II.")
endif()

# 4. Initialize and Project setup
deal_ii_initialize_cached_variables()
project(${TARGET} CXX)

# 5. Autopilot handles linking, flags (-O3), and Kokkos/MPI setup
deal_ii_invoke_autopilot()

# 6. Include Directories
target_include_directories(${TARGET} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
